from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class _Hook_amd64(Hook):
    """
    Implementation details for L{Hook} on the L{win32.ARCH_AMD64} architecture.
    """
    __new__ = object.__new__
    __float_types = (ctypes.c_double, ctypes.c_float)
    try:
        __float_types += (ctypes.c_longdouble,)
    except AttributeError:
        pass

    def _calc_signature(self, signature):
        self._cast_signature_pointers_to_void(signature)
        float_types = self.__float_types
        c_sizeof = ctypes.sizeof
        reg_size = c_sizeof(ctypes.c_size_t)
        reg_int_sig = []
        reg_float_sig = []
        stack_sig = []
        for i in compat.xrange(len(signature)):
            arg = signature[i]
            name = 'arg_%d' % i
            stack_sig.insert(0, (name, arg))
            if i < 4:
                if type(arg) in float_types:
                    reg_float_sig.append((name, arg))
                elif c_sizeof(arg) <= reg_size:
                    reg_int_sig.append((name, arg))
                else:
                    msg = "Hook signatures don't support structures within the first 4 arguments of a function for the %s architecture" % win32.arch
                    raise NotImplementedError(msg)
        if reg_int_sig:

            class RegisterArguments(ctypes.Structure):
                _fields_ = reg_int_sig
        else:
            RegisterArguments = None
        if reg_float_sig:

            class FloatArguments(ctypes.Structure):
                _fields_ = reg_float_sig
        else:
            FloatArguments = None
        if stack_sig:

            class StackArguments(ctypes.Structure):
                _fields_ = stack_sig
        else:
            StackArguments = None
        return (len(signature), RegisterArguments, FloatArguments, StackArguments)

    def _get_return_address(self, aProcess, aThread):
        return aProcess.read_pointer(aThread.get_sp())

    def _get_function_arguments(self, aProcess, aThread):
        if self._signature:
            args_count, RegisterArguments, FloatArguments, StackArguments = self._signature
            arguments = {}
            if StackArguments:
                address = aThread.get_sp() + win32.sizeof(win32.LPVOID)
                stack_struct = aProcess.read_structure(address, StackArguments)
                stack_args = dict([(name, stack_struct.__getattribute__(name)) for name, type in stack_struct._fields_])
                arguments.update(stack_args)
            flags = 0
            if RegisterArguments:
                flags = flags | win32.CONTEXT_INTEGER
            if FloatArguments:
                flags = flags | win32.CONTEXT_MMX_REGISTERS
            if flags:
                ctx = aThread.get_context(flags)
                if RegisterArguments:
                    buffer = (win32.QWORD * 4)(ctx['Rcx'], ctx['Rdx'], ctx['R8'], ctx['R9'])
                    reg_args = self._get_arguments_from_buffer(buffer, RegisterArguments)
                    arguments.update(reg_args)
                if FloatArguments:
                    buffer = (win32.M128A * 4)(ctx['XMM0'], ctx['XMM1'], ctx['XMM2'], ctx['XMM3'])
                    float_args = self._get_arguments_from_buffer(buffer, FloatArguments)
                    arguments.update(float_args)
            params = tuple([arguments['arg_%d' % i] for i in compat.xrange(args_count)])
        else:
            params = ()
        return params

    def _get_arguments_from_buffer(self, buffer, structure):
        b_ptr = ctypes.pointer(buffer)
        v_ptr = ctypes.cast(b_ptr, ctypes.c_void_p)
        s_ptr = ctypes.cast(v_ptr, ctypes.POINTER(structure))
        struct = s_ptr.contents
        return dict([(name, struct.__getattribute__(name)) for name, type in struct._fields_])

    def _get_return_value(self, aThread):
        ctx = aThread.get_context(win32.CONTEXT_INTEGER)
        return ctx['Rax']