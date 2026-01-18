import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
class _Kernel(serialize.ReduceMixin):
    """
    CUDA Kernel specialized for a given set of argument types. When called, this
    object launches the kernel on the device.
    """

    @global_compiler_lock
    def __init__(self, py_func, argtypes, link=None, debug=False, lineinfo=False, inline=False, fastmath=False, extensions=None, max_registers=None, opt=True, device=False):
        if device:
            raise RuntimeError('Cannot compile a device function as a kernel')
        super().__init__()
        self.objectmode = False
        self.entry_point = None
        self.py_func = py_func
        self.argtypes = argtypes
        self.debug = debug
        self.lineinfo = lineinfo
        self.extensions = extensions or []
        nvvm_options = {'fastmath': fastmath, 'opt': 3 if opt else 0}
        cc = get_current_device().compute_capability
        cres = compile_cuda(self.py_func, types.void, self.argtypes, debug=self.debug, lineinfo=lineinfo, inline=inline, fastmath=fastmath, nvvm_options=nvvm_options, cc=cc)
        tgt_ctx = cres.target_context
        code = self.py_func.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno
        lib, kernel = tgt_ctx.prepare_cuda_kernel(cres.library, cres.fndesc, debug, lineinfo, nvvm_options, filename, linenum, max_registers)
        if not link:
            link = []
        self.cooperative = 'cudaCGGetIntrinsicHandle' in lib.get_asm_str()
        if self.cooperative:
            lib.needs_cudadevrt = True
        res = [fn for fn in cuda_fp16_math_funcs if f'__numba_wrapper_{fn}' in lib.get_asm_str()]
        if res:
            basedir = os.path.dirname(os.path.abspath(__file__))
            functions_cu_path = os.path.join(basedir, 'cpp_function_wrappers.cu')
            link.append(functions_cu_path)
        for filepath in link:
            lib.add_linking_file(filepath)
        self.entry_name = kernel.name
        self.signature = cres.signature
        self._type_annotation = cres.type_annotation
        self._codelibrary = lib
        self.call_helper = cres.call_helper
        self.target_context = tgt_ctx
        self.fndesc = cres.fndesc
        self.environment = cres.environment
        self._referenced_environments = []
        self.lifted = []
        self.reload_init = []

    @property
    def library(self):
        return self._codelibrary

    @property
    def type_annotation(self):
        return self._type_annotation

    def _find_referenced_environments(self):
        return self._referenced_environments

    @property
    def codegen(self):
        return self.target_context.codegen()

    @property
    def argument_types(self):
        return tuple(self.signature.args)

    @classmethod
    def _rebuild(cls, cooperative, name, signature, codelibrary, debug, lineinfo, call_helper, extensions):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        super(cls, instance).__init__()
        instance.entry_point = None
        instance.cooperative = cooperative
        instance.entry_name = name
        instance.signature = signature
        instance._type_annotation = None
        instance._codelibrary = codelibrary
        instance.debug = debug
        instance.lineinfo = lineinfo
        instance.call_helper = call_helper
        instance.extensions = extensions
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
        return dict(cooperative=self.cooperative, name=self.entry_name, signature=self.signature, codelibrary=self._codelibrary, debug=self.debug, lineinfo=self.lineinfo, call_helper=self.call_helper, extensions=self.extensions)

    def bind(self):
        """
        Force binding to current CUDA context
        """
        self._codelibrary.get_cufunc()

    @property
    def regs_per_thread(self):
        """
        The number of registers used by each thread for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.regs

    @property
    def const_mem_size(self):
        """
        The amount of constant memory used by this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.const

    @property
    def shared_mem_per_block(self):
        """
        The amount of shared memory used per block for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.shared

    @property
    def max_threads_per_block(self):
        """
        The maximum allowable threads per block.
        """
        return self._codelibrary.get_cufunc().attrs.maxthreads

    @property
    def local_mem_per_thread(self):
        """
        The amount of local memory used per thread for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.local

    def inspect_llvm(self):
        """
        Returns the LLVM IR for this kernel.
        """
        return self._codelibrary.get_llvm_str()

    def inspect_asm(self, cc):
        """
        Returns the PTX code for this kernel.
        """
        return self._codelibrary.get_asm_str(cc=cc)

    def inspect_sass_cfg(self):
        """
        Returns the CFG of the SASS for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
        return self._codelibrary.get_sass_cfg()

    def inspect_sass(self):
        """
        Returns the SASS code for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
        return self._codelibrary.get_sass()

    def inspect_types(self, file=None):
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
        if self._type_annotation is None:
            raise ValueError('Type annotation is not available')
        if file is None:
            file = sys.stdout
        print('%s %s' % (self.entry_name, self.argument_types), file=file)
        print('-' * 80, file=file)
        print(self._type_annotation, file=file)
        print('=' * 80, file=file)

    def max_cooperative_grid_blocks(self, blockdim, dynsmemsize=0):
        """
        Calculates the maximum number of blocks that can be launched for this
        kernel in a cooperative grid in the current context, for the given block
        and dynamic shared memory sizes.

        :param blockdim: Block dimensions, either as a scalar for a 1D block, or
                         a tuple for 2D or 3D blocks.
        :param dynsmemsize: Dynamic shared memory size in bytes.
        :return: The maximum number of blocks in the grid.
        """
        ctx = get_context()
        cufunc = self._codelibrary.get_cufunc()
        if isinstance(blockdim, tuple):
            blockdim = functools.reduce(lambda x, y: x * y, blockdim)
        active_per_sm = ctx.get_active_blocks_per_multiprocessor(cufunc, blockdim, dynsmemsize)
        sm_count = ctx.device.MULTIPROCESSOR_COUNT
        return active_per_sm * sm_count

    def launch(self, args, griddim, blockdim, stream=0, sharedmem=0):
        cufunc = self._codelibrary.get_cufunc()
        if self.debug:
            excname = cufunc.name + '__errcode__'
            excmem, excsz = cufunc.module.get_global_symbol(excname)
            assert excsz == ctypes.sizeof(ctypes.c_int)
            excval = ctypes.c_int()
            excmem.memset(0, stream=stream)
        retr = []
        kernelargs = []
        for t, v in zip(self.argument_types, args):
            self._prepare_args(t, v, stream, retr, kernelargs)
        if driver.USE_NV_BINDING:
            zero_stream = driver.binding.CUstream(0)
        else:
            zero_stream = None
        stream_handle = stream and stream.handle or zero_stream
        driver.launch_kernel(cufunc.handle, *griddim, *blockdim, sharedmem, stream_handle, kernelargs, cooperative=self.cooperative)
        if self.debug:
            driver.device_to_host(ctypes.addressof(excval), excmem, excsz)
            if excval.value != 0:

                def load_symbol(name):
                    mem, sz = cufunc.module.get_global_symbol('%s__%s__' % (cufunc.name, name))
                    val = ctypes.c_int()
                    driver.device_to_host(ctypes.addressof(val), mem, sz)
                    return val.value
                tid = [load_symbol('tid' + i) for i in 'zyx']
                ctaid = [load_symbol('ctaid' + i) for i in 'zyx']
                code = excval.value
                exccls, exc_args, loc = self.call_helper.get_exception(code)
                if loc is None:
                    locinfo = ''
                else:
                    sym, filepath, lineno = loc
                    filepath = os.path.abspath(filepath)
                    locinfo = 'In function %r, file %s, line %s, ' % (sym, filepath, lineno)
                prefix = '%stid=%s ctaid=%s' % (locinfo, tid, ctaid)
                if exc_args:
                    exc_args = ('%s: %s' % (prefix, exc_args[0]),) + exc_args[1:]
                else:
                    exc_args = (prefix,)
                raise exccls(*exc_args)
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        for extension in reversed(self.extensions):
            ty, val = extension.prepare_args(ty, val, stream=stream, retr=retr)
        if isinstance(ty, types.Array):
            devary = wrap_arg(val).to_device(retr, stream)
            c_intp = ctypes.c_ssize_t
            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            ptr = driver.device_pointer(devary)
            if driver.USE_NV_BINDING:
                ptr = int(ptr)
            data = ctypes.c_void_p(ptr)
            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))
        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, 'c_%s' % ty)(val)
            kernelargs.append(cval)
        elif ty == types.float16:
            cval = ctypes.c_uint16(np.float16(val).view(np.uint16))
            kernelargs.append(cval)
        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)
        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)
        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)
        elif ty == types.complex64:
            kernelargs.append(ctypes.c_float(val.real))
            kernelargs.append(ctypes.c_float(val.imag))
        elif ty == types.complex128:
            kernelargs.append(ctypes.c_double(val.real))
            kernelargs.append(ctypes.c_double(val.imag))
        elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
            kernelargs.append(ctypes.c_int64(val.view(np.int64)))
        elif isinstance(ty, types.Record):
            devrec = wrap_arg(val).to_device(retr, stream)
            ptr = devrec.device_ctypes_pointer
            if driver.USE_NV_BINDING:
                ptr = ctypes.c_void_p(int(ptr))
            kernelargs.append(ptr)
        elif isinstance(ty, types.BaseTuple):
            assert len(ty) == len(val)
            for t, v in zip(ty, val):
                self._prepare_args(t, v, stream, retr, kernelargs)
        elif isinstance(ty, types.EnumMember):
            try:
                self._prepare_args(ty.dtype, val.value, stream, retr, kernelargs)
            except NotImplementedError:
                raise NotImplementedError(ty, val)
        else:
            raise NotImplementedError(ty, val)