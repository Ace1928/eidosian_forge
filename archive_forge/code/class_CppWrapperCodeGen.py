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
class CppWrapperCodeGen(WrapperCodeGen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        super().__init__()
        self.declare = 'auto '
        self.ending = ';'
        self.open_bracket = '{'
        self.closed_bracket = '}'
        self.comment = '//'
        self.namespace = 'at::'
        self.none_str = 'at::Tensor()'
        self.extern_call_ops = set()
        self.size = 'sizes()'
        self.stride = 'strides()'
        self.call_func_name = 'inductor_entry_cpp'
        self.cuda = False
        self.supports_intermediate_hooks = False
        self.outputs_need_copy = set()
        self.kernel_callsite_id = count()
        self.int_array_id = count()
        self.declared_int_array_vars = set()
        self.tmp_tensor_id = count()
        self.arg_var_id = count()
        self.used_cached_dtypes = set()
        from .cpp import cexpr, CppPrinter
        self.expr_printer = cexpr

        class GridExprCppPrinter(CppPrinter):

            def _print_FloorDiv(self, expr):
                x, div = expr.args
                x = self.paren(self.doprint(x))
                div = self.paren(self.doprint(div))
                assert expr.is_integer, 'Expect integers in GridExprPrinter'
                return f'({x}/{div})'
        self.grid_expr_printer = GridExprCppPrinter().doprint

    def generate_kernel_call(self, name, call_args, grid=None, device_index=None, cuda=True, triton=True):
        """
        Generates kernel call code.

        cuda: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if cuda:
            return super().generate_kernel_call(name, call_args, grid, device_index, cuda, triton)
        elif V.graph.aot_mode and config.aot_inductor.abi_compatible:
            from .cpp import DTYPE_TO_CPP
            new_args = []
            for arg in call_args:
                var_name = f'var_{next(self.arg_var_id)}'
                self.writeline(f'void *{var_name}{self.ending}')
                self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, &{var_name}));')
                dtype = V.graph.get_dtype(arg)
                cpp_dtype = DTYPE_TO_CPP[dtype]
                new_args.append(f'({cpp_dtype}*)({var_name})')
            self.writeline(self.wrap_kernel_call(name, new_args))
        else:
            self.writeline(self.wrap_kernel_call(name, call_args))

    def write_constant(self, name, hashed):
        self.header.writeline(f'// {name} {hashed}')

    def write_header(self):
        if V.graph.aot_mode:
            with open(os.path.join(os.path.dirname(__file__), 'aoti_runtime', 'interface.cpp')) as f:
                self.header.splice(f.read())
        else:
            self.header.splice("\n                import torch\n                from torch._inductor.codecache import CppWrapperCodeCache\n\n                cpp_wrapper_src = (\n                '''\n                ")
        if config.aot_inductor.abi_compatible:
            self.header.splice('#include <torch/csrc/inductor/aoti_torch/c/shim.h>')
        else:
            self.header.splice('\n                #include <ATen/ATen.h>\n                #include <ATen/core/dispatch/Dispatcher.h>\n                #include <ATen/native/BinaryOps.h>\n                #include <torch/csrc/inductor/aoti_torch/tensor_converter.h>\n                #include <torch/csrc/inductor/inductor_ops.h>\n                #define reinterpret_tensor torch::inductor::_reinterpret_tensor\n                #define alloc_from_pool torch::inductor::_alloc_from_pool\n                ')
        self.header.splice('#include <c10/util/generic_math.h>')
        from .memory_planning import ALIGN_BYTES
        self.header.splice(f'\n            [[maybe_unused]] static int64_t align(int64_t nbytes) {{\n              return (nbytes + {ALIGN_BYTES} - 1) & -{ALIGN_BYTES};\n            }}\n            ')

    def mark_output_type(self):
        from ..ir import ShapeAsConstantBuffer
        output_is_tensor = dict()
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True
        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        if V.graph.aot_mode:
            self.prefix.writeline('namespace torch {')
            self.prefix.writeline('namespace aot_inductor {')

    def write_input_output_info(self, info_kind: str, idx: int, name: str):
        self.prefix.writeline(f'{info_kind}[{idx}].name = "{name}";')

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            self.prefix.splice('\n                void AOTInductorModel::run_impl(\n                    AtenTensorHandle*\n                        input_handles, // array of input AtenTensorHandle; handles\n                                        // are stolen; the array itself is borrowed\n                    AtenTensorHandle*\n                        output_handles, // array for writing output AtenTensorHandle; handles\n                                        // will be stolen by the caller; the array itself is\n                                        // borrowed\n                    DeviceStreamType stream,\n                    AOTIProxyExecutorHandle proxy_executor\n                ) {\n                ')
        else:
            self.prefix.splice(f'std::vector<at::Tensor> {self.call_func_name}(const std::vector<at::Tensor>& inputs) {{')
        with self.prefix.indent():
            if V.graph.aot_mode:
                if config.aot_inductor.abi_compatible:
                    self.prefix.splice('\n                            auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, num_inputs());\n                        ')
                else:
                    self.prefix.splice('\n                            auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, num_inputs());\n                        ')
            else:
                self.prefix.splice('\n                        py::gil_scoped_release release;\n                    ')
            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype
                        from .cpp import DTYPE_TO_CPP
                        dtype = may_get_constant_buffer_dtype(V.graph.graph_inputs[input_key])
                        assert dtype is not None, 'Fails to get the dtype of the sympy.Expr'
                        cpp_dtype = DTYPE_TO_CPP[dtype]
                        assert not config.aot_inductor.abi_compatible, 'Need to add .item support for abi_compatible AOTInductor codegen'
                        self.prefix.writeline(f'{cpp_dtype} {input_key} = inputs[{idx}].item<{cpp_dtype}>();')
                    else:
                        self.prefix.writeline(f'auto {input_key} = std::move(inputs[{idx}]);')
            assert all((isinstance(v, torch.Tensor) for v in list(V.graph.constants.values()))), 'Expect all constants to be Tensor'
            for idx, constants_key in enumerate(V.graph.constants.keys()):
                if V.graph.aot_mode:
                    if config.aot_inductor.abi_compatible:
                        self.prefix.writeline(f'auto {constants_key} = constants_.at({idx});')
                    else:
                        self.prefix.writeline(f'auto {constants_key} = *tensor_handle_to_tensor_pointer(' + f'constants_.at({idx}));')
                else:
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(f'auto {constants_key} = inputs[{constants_idx}];')
            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if V.graph.aot_mode:
                self.prefix.writeline('inputs.clear();')
                self.prefix.writeline('auto& kernels = *dynamic_cast<AOTInductorModelKernels*>(this->kernels_.get());')

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        if config.aot_inductor.abi_compatible:
            code.writeline(f'int64_t* {name}_size;')
            code.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes({name}, &{name}_size));')
        else:
            super().codegen_input_size_var_decl(code, name)

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        if config.aot_inductor.abi_compatible:
            code.writeline(f'int64_t* {name}_stride;')
            code.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides({name}, &{name}_stride));')
        else:
            super().codegen_input_stride_var_decl(code, name)

    def codegen_model_kernels(self):
        self.prefix.writeline('namespace {')
        self.prefix.writeline('class AOTInductorModelKernels : public AOTInductorModelKernelsBase {')
        self.prefix.writeline('  public:')
        for kernel in chain(self.src_to_kernel.values(), self.user_defined_kernel_cache.values()):
            self.prefix.writeline(f'    CUfunction {kernel}{{nullptr}};')
        self.prefix.writeline('};')
        self.prefix.writeline('}  // namespace')

    def codegen_model_constructor(self):
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """
        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        self.prefix.splice(f'\n            AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map, std::optional<std::string> cubin_dir)\n                : AOTInductorModelBase({num_inputs}, {num_outputs}, {num_constants}, cubin_dir) {{\n            ')
        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                assert not isinstance(inp, sympy.Expr), f'input name={name!r} cannot be symbolic'
                self.write_input_output_info('inputs_info_', idx, name)
            for idx, (name, tensor) in enumerate(V.graph.constants.items()):
                assert isinstance(tensor, torch.Tensor)
                self.prefix.writeline(f'constants_info_[{idx}].name = "{name}";')
                self.prefix.writeline(f'constants_info_[{idx}].dtype = static_cast<int32_t>({self.codegen_dtype(tensor.dtype)});')
                self.prefix.writeline(f'constants_info_[{idx}].offset = {tensor.storage_offset()};')
                self.prefix.writeline(f'constants_info_[{idx}].data_size = {tensor.untyped_storage().nbytes()};')
                size_str = ', '.join([str(s) for s in tensor.size()])
                self.prefix.writeline(f'constants_info_[{idx}].shape = {{{size_str}}};')
                stride_str = ', '.join([str(s) for s in tensor.stride()])
                self.prefix.writeline(f'constants_info_[{idx}].stride = {{{stride_str}}};')
            self.prefix.writeline('update_constants_map(std::move(constants_map));')

            def escape_string(x):
                return x.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
            self.prefix.writeline(f'in_spec_ = "{escape_string(config.aot_inductor.serialized_in_spec)}";')
            self.prefix.writeline(f'out_spec_ = "{escape_string(config.aot_inductor.serialized_out_spec)}";')
            for idx, output in enumerate(V.graph.graph_outputs):
                assert not isinstance(output, sympy.Expr), f'output name={name!r} cannot be symbolic'
                name = f'output{idx}'
                self.write_input_output_info('outputs_info_', idx, name)
            self.prefix.writeline('this->kernels_ = std::make_unique<AOTInductorModelKernels>();')
        self.prefix.writeline('}')

    def generate(self, is_inference):
        if V.graph.aot_mode:
            self.codegen_model_kernels()
            self.codegen_model_constructor()
        self.write_wrapper_decl()
        return super().generate(is_inference)

    def finalize_prefix(self):
        cached_dtypes_buffer = IndentedBuffer()
        if config.aot_inductor.abi_compatible:
            for dtype in self.used_cached_dtypes:
                cached_dtypes_buffer.writeline(f'CACHE_TORCH_DTYPE({dtype});')
        cached_dtypes_buffer.splice(self.prefix)
        self.prefix = cached_dtypes_buffer

    def define_kernel(self, name: str, kernel: str, metadata: Optional[str]=None, cuda=False):
        self.header.splice(f'\n{kernel}\n')

    def generate_return(self, output_refs):
        if V.graph.aot_mode:
            cst_names = V.graph.constants.keys()
            for idx, output in enumerate(output_refs):
                if output in cst_names:
                    if config.aot_inductor.abi_compatible:
                        self.wrapper_call.writeline(f'aoti_torch_clone({output}, &output_handles[{idx}]);')
                    else:
                        self.wrapper_call.writeline(f'output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>(' + f'new at::Tensor(std::move({output}.clone())));')
                elif config.aot_inductor.abi_compatible:
                    if output in self.cached_thread_locals:
                        self.wrapper_call.writeline(f'aoti_torch_new_uninitialized_tensor(&output_handles[{idx}]);')
                        self.wrapper_call.writeline(f'aoti_torch_assign_tensors({output}, output_handles[{idx}]);')
                    else:
                        self.wrapper_call.writeline(f'output_handles[{idx}] = {output}.release();')
                else:
                    self.wrapper_call.writeline(f'output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>(' + f'new at::Tensor({output}));')
        else:
            self.wrapper_call.writeline(f'return {{{', '.join(output_refs)}}};\n}}')

    def generate_end(self, result):
        if V.graph.aot_mode:
            result.writeline('} // AOTInductorModel::run_impl')
            result.writeline('} // namespace aot_inductor')
            result.writeline('} // namespace torch')
            return
        result.writeline("'''\n)")
        wrapper_call_hash = codecache.code_hash(result.getvalue())
        result.splice(f"\n            module = CppWrapperCodeCache.load(cpp_wrapper_src, '{self.call_func_name}', '{wrapper_call_hash}', {self.cuda})\n            ")
        if all((x for x in self.output_is_tensor.values())):
            return_str = 'return f(args_tensor)'
        else:
            outputs = [f'outputs[{i}]' if self.output_is_tensor[i] else f'outputs[{i}].item()' for i in range(len(V.graph.graph_outputs))]
            outputs_str = f'[{', '.join(outputs)}]'
            return_str = f'\n                    outputs = f(args_tensor)\n                    return {outputs_str}\n            '
        args_str = 'args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]'
        if V.graph.constants:
            assert all((isinstance(v, torch.Tensor) for v in list(V.graph.constants.values()))), 'Expect all constants to be Tensor'
            constants_str = f'[{', '.join(V.graph.constants.keys())}]'
            args_str += f'\n                    constants_tensor = {constants_str}\n                    args_tensor.extend(constants_tensor)\n            '
        result.splice(f'\n            def _wrap_func(f):\n                def g(args):\n                    {args_str}\n                    {return_str}\n                return g\n            call = _wrap_func(module.{self.call_func_name})\n            ')

    def generate_c_shim_extern_kernel_call(self, kernel, args):
        kernel_tokens = kernel.split('::')
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == 'call':
            kernel_suffix = kernel_tokens[-2]
        shim_fn = f'aoti_torch_{kernel_suffix}'
        self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(args)}));')

    def generate_c_shim_extern_kernel_alloc(self, extern_kernel, args):
        name = extern_kernel.name
        output_handle_name = f'{name}_handle'
        self.writeline(f'AtenTensorHandle {output_handle_name};')
        output_arg = f'&{output_handle_name}'
        self.generate_c_shim_extern_kernel_call(extern_kernel.codegen_kernel_name(), args + [output_arg])
        self.writeline(f'RAIIAtenTensorHandle {name}({output_handle_name});')

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)
        else:
            super().generate_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(self, fallback_kernel, args):
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                name = f'{output.get_name()}'
                output_handle_name = f'{name}_handle'
                if output.indices:
                    assert output.indices[0][1] == idx, f'expected output.indices[0][1]={output.indices[0][1]!r} == idx={idx!r} for output_name_base={output_name_base!r}'
                self.writeline(f'AtenTensorHandle {output_handle_name};')
                output_args.append(f'&{output_handle_name}')
                output_raii_handles.append(f'RAIIAtenTensorHandle {name}({output_handle_name});')
            elif isinstance(output, int):
                output_name = f'{output_name_base}_{idx}'
                self.writeline(f'int64_t {output_name} = {output};')
                output_args.append(f'&{output_name}')
            elif output is None:
                output_args.append('nullptr')
            else:
                raise NotImplementedError('unsupported type of {output=}')
        args = args + output_args
        assert fallback_kernel.abi_compatible_kernel is not None, f'abi_compatible_kernel is None for fallback_kernel.kernel={fallback_kernel.kernel!r}'
        self.generate_c_shim_extern_kernel_call(fallback_kernel.abi_compatible_kernel, args)
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def generate_fallback_kernel(self, fallback_kernel, args):
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_fallback_kernel(fallback_kernel, args)
        else:
            super().generate_fallback_kernel(fallback_kernel, args)

    def generate_extern_kernel_out(self, output_view, codegen_reference, args, kernel):
        if output_view:
            output_as_strided = f'{output_view.codegen_reference()}'
            output_name = f'{output_view.get_name()}_as_strided'
            self.writeline(f'auto {output_name} = {output_as_strided};')
            args.insert(0, output_name)
        else:
            args.insert(0, f'{codegen_reference}')
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_extern_kernel_call(kernel, args)
        else:
            self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_user_defined_triton_kernel(self, kernel_name, grid, configs, args):
        assert len(grid) != 0
        if len(grid) == 1:
            grid_decision = grid[0]
        else:
            meta = CudaKernelParamCache.get(kernel_name)
            assert meta is not None
            grid_decision = None
            for i, c in enumerate(configs):
                if all((arg == meta['meta'][key] for key, arg in c.kwargs.items())):
                    grid_decision = grid[i]
                    break
            assert grid_decision is not None
        self.generate_kernel_call(kernel_name, args, grid=grid_decision, device_index=V.graph.scheduler.current_device.index, cuda=True, triton=True)

    def generate_scatter_fallback(self, output, inputs, kernel, fn, src_is_tensor, reduce, kwargs):
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            kernel = kernel.replace('at::', 'aoti_torch_')
        line = f'{kernel}({output}, {','.join(map(str, inputs))}'
        if fn == 'aten.scatter_':
            if src_is_tensor:
                if reduce:
                    line += f', {V.graph.wrapper_code.val_to_arg_str(reduce)}'
            else:
                assert reduce is None, 'Expect reduce to be None for aten.scatter_ with scalar src'
        else:
            line += f', {','.join(kwargs)}'
        line += f'){self.ending}'
        self.writeline(line)

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.expr_printer(V.graph.sizevars.simplify(x))

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            return name
        else:
            return f'std::get<{index}>({basename})'

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return '{}'
        if len(parts) == 1:
            return f'{{{parts[0]}, }}'
        return f'{{{', '.join(parts)}}}'

    def is_statically_known_int(self, x):
        try:
            val = V.graph._shape_env._maybe_evaluate_static(x)
            int(x)
            return True
        except Exception:
            return False

    def is_statically_known_list_of_ints(self, lst):
        return all((isinstance(self.is_statically_known_int(x), int) for x in lst))

    def can_prove_buffer_has_static_shape(self, buffer):
        return self.is_statically_known_list_of_ints(buffer.get_size())

    def can_cache_buffer_in_thread_local(self, buffer):
        return not self.cuda and config.allow_buffer_reuse and self.can_prove_buffer_has_static_shape(buffer)

    def make_buffer_free(self, buffer):
        return '' if isinstance(buffer.get_layout(), ir.MultiOutputLayout) or (V.graph.aot_mode and self.can_cache_buffer_in_thread_local(buffer)) else f'{buffer.get_name()}.reset();'

    def make_free_by_names(self, names_to_del: List[str]):
        return ' '.join((f'{name}.reset();' for name in names_to_del))

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        if config.aot_inductor.abi_compatible:
            return f'auto {new_name} = std::move({old_name});  // reuse'
        else:
            return super().codegen_exact_buffer_reuse(old_name, new_name, del_line)

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline('RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>());')

    def write_triton_header_once(self):
        pass

    def generate_start_graph(self):
        pass

    def generate_end_graph(self):
        pass

    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_check_inf_and_nan({buf}));')

    def codegen_device(self, device):
        if config.aot_inductor.abi_compatible:
            return f'cached_torch_device_type_{device.type},{(device.index if device.index else 0)}'
        else:
            from .cpp import DEVICE_TO_ATEN
            return f'c10::Device({DEVICE_TO_ATEN[device.type]}, {device.index})' if device.index is not None else f'{DEVICE_TO_ATEN[device.type]}'

    def codegen_dtype(self, dtype):
        if config.aot_inductor.abi_compatible:
            dtype_str = str(dtype).split('.')[-1]
            self.used_cached_dtypes.add(dtype_str)
            return f'cached_torch_dtype_{dtype_str}'
        else:
            from .cpp import DTYPE_TO_ATEN
            return DTYPE_TO_ATEN[dtype]

    @functools.lru_cache(None)
    def codegen_int_array_var(self, int_array: str, writer=None):
        if writer is None:
            writer = self
        var = f'int_array_{next(self.int_array_id)}'
        if var not in self.declared_int_array_vars:
            self.declared_int_array_vars.add(var)
            writer.writeline(f'int64_t {var}[] = {int_array};')
        return var

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(buffer.get_name(), buffer.get_device(), buffer.get_dtype(), buffer.get_size(), buffer.get_stride(), self.can_cache_buffer_in_thread_local(buffer))

    def make_allocation(self, name, device, dtype, shape, stride, can_cache_buffer_in_thread_local=False):
        device = self.codegen_device(device)
        dtype = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        if config.aot_inductor.abi_compatible:
            device_type, device_id = device.split(',')
            args = [str(len(shape)), self.codegen_int_array_var(size, self.wrapper_call), self.codegen_int_array_var(stride, self.wrapper_call), dtype, device_type, 'this->device_idx_' if V.graph.aot_mode else device_id, f'&{name}_handle']

            def gen_alloc(wrapper_call, name, args):
                wrapper_call.writeline(f'AtenTensorHandle {name}_handle;')
                wrapper_call.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));')
            if can_cache_buffer_in_thread_local:
                self.cached_thread_locals.add(name)
                self.wrapper_call.writeline(f'thread_local RAIIAtenTensorHandle {name}_handle = ([&] {{')
                with self.wrapper_call.indent():
                    gen_alloc(self.wrapper_call, name, args)
                    self.wrapper_call.writeline(f'return {name}_handle;')
                self.wrapper_call.writeline('})();')
                return f'AtenTensorHandle {name}({name}_handle.get());'
            else:
                gen_alloc(self.wrapper_call, name, args)
                return f'RAIIAtenTensorHandle {name}({name}_handle);'
        if V.graph.aot_mode and device.startswith('c10::Device('):
            tensor_device = f'{device.split(',')[0]}, this->device_idx_)'
        else:
            tensor_device = device
        return f'{self.declare}{name} = {self.namespace}empty_strided({size}, {stride}, at::TensorOptions({tensor_device}).dtype({dtype})){self.ending}'

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        if config.aot_inductor.abi_compatible:
            size = self.codegen_shape_tuple(shape)
            stride = self.codegen_shape_tuple(stride)
            tmp_name = f'tmp_tensor_handle_{next(self.tmp_tensor_id)}'
            args = [name, pexpr(offset), self.codegen_dtype(dtype), str(len(shape)), self.codegen_int_array_var(size, self.wrapper_call), self.codegen_int_array_var(stride, self.wrapper_call), f'&{tmp_name}']
            self.wrapper_call.writeline(f'AtenTensorHandle {tmp_name};')
            self.wrapper_call.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));')
            return f'RAIIAtenTensorHandle({tmp_name})'
        return 'alloc_from_pool({})'.format(', '.join([name, pexpr(offset), self.codegen_dtype(dtype), self.codegen_shape_tuple(shape), self.codegen_shape_tuple(stride)]))

    def codegen_reinterpret_view(self, data, size_list, stride_list, offset, writer) -> str:
        dim = str(len(size_list))
        size = self.codegen_shape_tuple(size_list)
        stride = self.codegen_shape_tuple(stride_list)
        offset = self.codegen_sizevar(offset)
        if config.aot_inductor.abi_compatible:
            tmp_name = f'tmp_tensor_handle_{next(self.tmp_tensor_id)}'
            if writer is None:
                writer = self
            args = [f'{data.get_name()}', dim, self.codegen_int_array_var(size, writer), self.codegen_int_array_var(stride, writer), offset, f'&{tmp_name}']

            def gen_reinterpret_call(writer, args):
                writer.writeline(f'AtenTensorHandle {tmp_name};')
                writer.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor({', '.join(args)}));')
            if self.can_cache_buffer_in_thread_local(data) and self.is_statically_known_list_of_ints(size_list) and self.is_statically_known_list_of_ints(stride_list):
                self.cached_thread_locals.add(tmp_name)
                writer.writeline(f'thread_local RAIIAtenTensorHandle {tmp_name}_handle = ([&] {{')
                if hasattr(writer, 'indent'):
                    indent = writer.indent()
                else:
                    indent = contextlib.nullcontext()
                with indent:
                    gen_reinterpret_call(writer, args)
                    writer.writeline(f'return {tmp_name};')
                writer.writeline('})();')
                writer.writeline(f'AtenTensorHandle {tmp_name}({tmp_name}_handle.get());')
                return tmp_name
            gen_reinterpret_call(writer, args)
            return f'RAIIAtenTensorHandle({tmp_name})'
        else:
            args = [data.get_name(), size, stride, offset]
            return f'reinterpret_tensor({', '.join(args)})'

    def codegen_device_copy(self, src, dst):
        if config.aot_inductor.abi_compatible:
            self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_tensor_copy_({src}, {dst}));')
        else:
            self.writeline(f'{dst}.copy_({src});')

    def codegen_multi_output(self, name, value):
        if not config.aot_inductor.abi_compatible:
            super().codegen_multi_output(name, value)

    def generate_extern_kernel_args_decl_if_needed(self, op_overload, raw_args, output_args):
        arg_types = [x.real_type for x in op_overload._schema.arguments]
        return_types = [x.type for x in op_overload._schema.returns]
        new_tensor_args = []
        new_int_args = []

        def fill_args(arg, arg_type):
            static_arg_types = (torch.FloatType, torch.BoolType, torch.StringType, torch.Type, torch.DeviceObjType)
            inductor_tensor_buffers = (ir.Buffer, ir.ReinterpretView)
            if isinstance(arg_type, torch.TensorType):
                assert isinstance(arg, inductor_tensor_buffers), f'got {type(arg)}'
                new_tensor_args.append(f'{arg.codegen_reference()}')
            elif isinstance(arg_type, torch.IntType):
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.SymIntType):
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.NumberType):
                assert isinstance(arg, (int, float, bool))
                if isinstance(arg, int):
                    new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.ListType):
                assert isinstance(arg, (list, tuple))
                if isinstance(arg_type.getElementType(), torch.TensorType):
                    new_tensor_args.extend([f'{a.codegen_reference()}' for a in arg])
                elif isinstance(arg_type.getElementType(), torch.OptionalType) and isinstance(arg_type.getElementType().getElementType(), torch.TensorType):
                    new_tensor_args.extend([f'{a.codegen_reference()}' for a in arg if a is not None])
                elif isinstance(arg_type.getElementType(), (torch.IntType, torch.SymIntType)):
                    new_int_args.extend([str(a) for a in arg])
                elif isinstance(arg_type.getElementType(), torch.NumberType):
                    is_int_type = [isinstance(a, int) for a in arg]
                    if any(is_int_type):
                        assert all(is_int_type), 'AOTInductor only supports int scalars of the same type'
                        new_int_args.extend([str(a) for a in arg])
                else:
                    assert isinstance(arg_type.getElementType(), static_arg_types), f'Fall through arguments must be one of static_arg_types, got {type(arg_type)}'
            else:
                assert isinstance(arg_type, static_arg_types), f'Fall through arguments must be one of static_arg_types, got {type(arg_type)}'
        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(arg, return_type):
            if isinstance(return_type, torch.TensorType):
                self.writeline(f'AtenTensorHandle {arg}_handle;  // output buffer')
                self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));')
                self.writeline(f'RAIIAtenTensorHandle {arg}({arg}_handle);')
                new_tensor_args.append(f'{arg}')
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError('NYI support for return type: SymInt')
            elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.SymIntType):
                raise NotImplementedError('NYI support for return type: List[SymInt]')
            else:
                raise AssertionError(f'Unsupported return type found: {return_type}')
        for return_type in return_types:
            if isinstance(return_type, torch.TensorType):
                pass
            elif isinstance(return_type, torch.OptionalType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            elif isinstance(return_type, torch.ListType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            else:
                raise NotImplementedError(f'return type {return_type} is not yet supported.')
        for output_arg in output_args:
            assert output_arg is not None, 'Optional return types are not yet supported'
            if isinstance(output_arg, (list, tuple)):
                for out in output_arg:
                    fill_output_arg(out, torch.TensorType.get())
            else:
                fill_output_arg(output_arg, torch.TensorType.get())
        return (new_tensor_args, new_int_args)

    def generate_extern_kernel_alloc_and_find_schema_if_needed(self, name, kernel, codegen_args, cpp_op_schema, cpp_kernel_key, cpp_kernel_overload_name='', op_overload=None, raw_args=None, outputs=None):
        if config.is_fbcode():
            assert op_overload is not None
            assert raw_args is not None
            assert outputs is not None
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(name, cpp_kernel_key, op_overload, raw_args, outputs)
        else:
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_oss(name, kernel, codegen_args, cpp_op_schema, cpp_kernel_key, cpp_kernel_overload_name)

    def generate_extern_kernel_alloc_and_find_schema_if_needed_oss(self, name, kernel, codegen_args, cpp_op_schema, cpp_kernel_key, cpp_kernel_overload_name=''):
        if cpp_kernel_key not in self.extern_call_ops:
            self.writeline(f'static auto op_{cpp_kernel_key} = c10::Dispatcher::singleton()')
            self.writeline(f'\t.findSchemaOrThrow("{kernel}", "{cpp_kernel_overload_name}")')
            self.writeline(f'\t.typed<{cpp_op_schema}>();')
            self.extern_call_ops.add(cpp_kernel_key)
        self.writeline(f'auto {name} = op_{cpp_kernel_key}.call({', '.join(codegen_args)});')

    def generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(self, name, cpp_kernel_key, op_overload, raw_args, outputs):

        def extract_output_name(out):
            assert out is not None, 'None, i.e. optional output is not supported'
            if isinstance(out, ir.MultiOutput):
                return out.get_name()
            elif isinstance(out, (list, tuple)):
                return type(out)((extract_output_name(o) for o in out))
            else:
                raise AssertionError(f'Unexpected output: {type(out)}')
        output_args = extract_output_name(outputs)
        if isinstance(output_args, str):
            output_args = [output_args]
        tensor_call_args, int_call_args = self.generate_extern_kernel_args_decl_if_needed(op_overload, raw_args, output_args)
        tensor_call_args_str = ', '.join(tensor_call_args)
        int_call_args_str = ', '.join(int_call_args)
        extern_kernel_node_index = len(V.graph.extern_kernel_nodes) - 1
        self.writeline(f'aoti_torch_proxy_executor_call_function(proxy_executor, {extern_kernel_node_index}, {len(int_call_args)}, std::vector<int64_t>{{{int_call_args_str}}}.data(), {len(tensor_call_args)}, std::vector<AtenTensorHandle>{{{tensor_call_args_str}}}.data());')
        self.extern_call_ops.add(cpp_kernel_key)

    def val_to_cpp_arg_str(self, type_, val, is_legacy_abi) -> str:
        if config.aot_inductor.abi_compatible and (not is_legacy_abi) and isinstance(type_, torch.OptionalType):
            if val is None:
                return '0'
            if isinstance(val, (bool, int, str, float)):
                var_name = f'var_{next(self.arg_var_id)}'
                self.writeline(f'auto {var_name} = {self.val_to_arg_str(val)};')
                return f'&{var_name}'
            if not isinstance(type_.getElementType(), torch.TensorType):
                return f'&{self.val_to_arg_str(val)}'
        return self.val_to_arg_str(val)

    def val_to_arg_str(self, val) -> str:
        if val is None:
            if config.aot_inductor.abi_compatible:
                return '0'
            return 'c10::nullopt'
        elif isinstance(val, bool):
            if config.aot_inductor.abi_compatible:
                return '1' if val else '0'
            else:
                return 'true' if val else 'false'
        elif isinstance(val, int):
            return f'{val}L'
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, (ComputedBuffer, InputBuffer, ReinterpretView)):
            return val.codegen_reference()
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        elif isinstance(val, float) and val in [float('inf'), float('-inf')]:
            if val == float('inf'):
                return 'std::numeric_limits<float>::infinity()'
            else:
                return '-std::numeric_limits<float>::infinity()'
        elif isinstance(val, (list, tuple)):
            result = f'{{{', '.join((self.val_to_arg_str(x) for x in val))}}}'
            if config.aot_inductor.abi_compatible:
                return f'{self.codegen_int_array_var(result)}, {len(val)}'
            else:
                return result
        else:
            return repr(val)