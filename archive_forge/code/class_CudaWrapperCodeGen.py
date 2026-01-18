import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
class CudaWrapperCodeGen(CppWrapperCodeGen):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self):
        super().__init__()
        self.grid_id = count()
        self.cuda = True

    def write_header(self):
        super().write_header()
        self.header.splice('#include <filesystem>')
        if not config.aot_inductor.abi_compatible:
            self.header.splice('\n                #include <c10/cuda/CUDAGuard.h>\n                #include <c10/cuda/CUDAStream.h>\n                ')
        self.header.splice('\n            #define CUDA_DRIVER_CHECK(EXPR)                    \\\n            do {                                               \\\n                CUresult code = EXPR;                          \\\n                const char *msg;                               \\\n                cuGetErrorString(code, &msg);                  \\\n                if (code != CUDA_SUCCESS) {                    \\\n                    throw std::runtime_error(                  \\\n                        std::string("CUDA driver error: ") +   \\\n                        std::string(msg));                     \\\n                }                                              \\\n            } while (0);\n\n            namespace {\n\n            struct Grid {\n                Grid(uint32_t x, uint32_t y, uint32_t z)\n                  : grid_x(x), grid_y(y), grid_z(z) {}\n                uint32_t grid_x;\n                uint32_t grid_y;\n                uint32_t grid_z;\n\n                bool is_non_zero() {\n                    return grid_x > 0 && grid_y > 0 && grid_z > 0;\n                }\n            };\n\n            }  // anonymous namespace\n\n            static inline CUfunction loadKernel(\n                    std::string filePath,\n                    const std::string &funcName,\n                    uint32_t sharedMemBytes,\n                    const std::optional<std::string> &cubinDir = std::nullopt) {\n                if (cubinDir) {\n                    std::filesystem::path p1{*cubinDir};\n                    std::filesystem::path p2{filePath};\n                    filePath = (p1 / p2.filename()).string();\n                }\n\n                CUmodule mod;\n                CUfunction func;\n                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));\n                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));\n                if (sharedMemBytes > 0) {\n                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(\n                        func,\n                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,\n                        sharedMemBytes\n                    ))\n                }\n                return func;\n            }\n\n            static inline void launchKernel(\n                    CUfunction func,\n                    uint32_t gridX,\n                    uint32_t gridY,\n                    uint32_t gridZ,\n                    uint32_t numWarps,\n                    uint32_t sharedMemBytes,\n                    void* args[],\n                    cudaStream_t stream) {\n                CUDA_DRIVER_CHECK(cuLaunchKernel(\n                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr\n                ));\n            }\n            ')

    def write_get_raw_stream(self, index):
        name = f'stream{index}'
        self.writeline(f'cudaStream_t {name} = at::cuda::getCurrentCUDAStream({index});')
        return name

    def define_kernel(self, name: str, kernel: str, metadata: Optional[str]=None, cuda=True):
        if not cuda:
            return super().define_kernel(name, kernel, metadata, cuda)

    def generate(self, is_inference):
        self.prefix.writeline('\n')
        if not V.graph.aot_mode:
            for kernel in chain(self.src_to_kernel.values(), self.user_defined_kernel_cache.values()):
                self.prefix.writeline(f'static CUfunction {kernel} = nullptr;')
            self.prefix.writeline('\n')
        return super().generate(is_inference)

    @functools.lru_cache(None)
    def generate_load_kernel_once(self, name: str, mangled_name: str, cubin_path: str, shared_mem: int):
        if V.graph.aot_mode:
            self.writeline(f'if (kernels.{name} == nullptr) {{')
            self.writeline(f'    kernels.{name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem}, this->cubin_dir_);')
            self.writeline('}')
        else:
            self.writeline(f'if ({name} == nullptr) {{')
            self.writeline(f'    {name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem});')
            self.writeline('}')

    def generate_args_decl(self, call_args):
        dynamic_symbols = V.graph.sizevars.free_symbols()
        new_args = []
        for arg in call_args:
            var_name = f'var_{next(self.arg_var_id)}'
            if isinstance(arg, (sympy.Integer, sympy.Symbol, SymbolicCallArg)):
                self.writeline(f'auto {var_name} = {arg};')
            elif isinstance(arg, sympy.Expr):
                self.writeline(f'auto {var_name} = {self.expr_printer(arg)};')
            elif is_int(arg):
                self.writeline(f'int {var_name} = {arg};')
            elif is_float(arg):
                self.writeline(f'float {var_name} = {arg};')
            elif any((str(arg) == s.name for s in dynamic_symbols)):
                self.writeline(f'auto {var_name} = {arg};')
            elif arg == 'nullptr':
                self.writeline(f'auto {var_name} = nullptr;')
            elif arg == 'c10::nullopt':
                self.writeline(f'auto {var_name} = c10::nullopt;')
            elif config.aot_inductor.abi_compatible:
                self.writeline(f'CUdeviceptr {var_name};')
                self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, reinterpret_cast<void**>(&{var_name})));')
            else:
                self.writeline(f'CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());')
            new_args.append(f'&{var_name}')
        return ', '.join(new_args)

    def generate_default_grid(self, name: str, grid: List[Any], cuda: bool=True):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics.
        """
        if not cuda:
            return grid
        assert isinstance(grid, list), f'expected grid={grid!r} to be a list'
        grid = [e.inner_expr if isinstance(e, SymbolicCallArg) else e for e in grid]
        grid_fn = default_grid(*grid)
        params = CudaKernelParamCache.get(name)
        assert params is not None, f'cuda kernel parameters for {name} should already exist at this moment'
        block_cfg = {'XBLOCK': params['x_block'], 'YBLOCK': params['y_block'], 'ZBLOCK': params['z_block']}
        return grid_fn(block_cfg)

    def generate_kernel_call(self, name, call_args, grid=None, device_index=None, cuda=True, triton=True):
        if not cuda:
            return super().generate_kernel_call(name, call_args, grid, device_index, cuda, triton)
        params = CudaKernelParamCache.get(name)
        assert params is not None, f'cuda kernel parameters for {name} should already exist at this moment'
        mangled_name = params.get('mangled_name', None)
        assert mangled_name is not None, 'missing mangled_name'
        cubin_path = params.get(get_cpp_wrapper_cubin_path_name(), None)
        assert cubin_path is not None and os.path.exists(cubin_path), f'cubin file should already exist at this moment: {cubin_path}'
        shared_mem = params.get('shared_mem', 0)
        self.generate_load_kernel_once(name, mangled_name, cubin_path, shared_mem)
        call_args = self.generate_args_decl(call_args)
        kernel_args_var = f'kernel_args_var_{next(self.kernel_callsite_id)}'
        self.writeline(f'void* {kernel_args_var}[] = {{{call_args}}};')
        stream = 'stream' if V.graph.aot_mode else self.write_get_raw_stream(device_index)
        grid_name = f'{name}_grid_{next(self.grid_id)}'
        assert isinstance(grid, (list, tuple)), f'expected grid to be a list or tuple but got: grid={grid!r}'
        grid = [V.graph.sizevars.simplify(item) for item in grid]
        grid_has_unbacked_symbols = any((free_unbacked_symbols(item) for item in grid))
        grid_args = [self.grid_expr_printer(item) for item in grid]
        grid_args_str = ', '.join(grid_args)
        self.writeline(f'Grid {grid_name} = Grid({grid_args_str});')
        if grid_has_unbacked_symbols:
            self.writeline(f'if ({grid_name}.is_non_zero()) {{')
        kernel_var_name = f'kernels.{name}' if V.graph.aot_mode else name
        self.writeline('launchKernel({}, {}, {}, {}, {}, {}, {}, {});'.format(kernel_var_name, f'{grid_name}.grid_x', f'{grid_name}.grid_y', f'{grid_name}.grid_z', params['num_warps'], params['shared_mem'], kernel_args_var, stream))
        if grid_has_unbacked_symbols:
            self.writeline('}')