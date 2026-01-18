from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class ExceptionEvent(Event):
    """
    Exception event.

    @type exceptionName: dict( int S{->} str )
    @cvar exceptionName:
        Mapping of exception constants to their names.

    @type exceptionDescription: dict( int S{->} str )
    @cvar exceptionDescription:
        Mapping of exception constants to user-friendly strings.

    @type breakpoint: L{Breakpoint}
    @ivar breakpoint:
        If the exception was caused by one of our breakpoints, this member
        contains a reference to the breakpoint object. Otherwise it's not
        defined. It should only be used from the condition or action callback
        routines, instead of the event handler.

    @type hook: L{Hook}
    @ivar hook:
        If the exception was caused by a function hook, this member contains a
        reference to the hook object. Otherwise it's not defined. It should
        only be used from the hook callback routines, instead of the event
        handler.
    """
    eventName = 'Exception event'
    eventDescription = 'An exception was raised by the debugee.'
    __exceptionMethod = {win32.EXCEPTION_ACCESS_VIOLATION: 'access_violation', win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED: 'array_bounds_exceeded', win32.EXCEPTION_BREAKPOINT: 'breakpoint', win32.EXCEPTION_DATATYPE_MISALIGNMENT: 'datatype_misalignment', win32.EXCEPTION_FLT_DENORMAL_OPERAND: 'float_denormal_operand', win32.EXCEPTION_FLT_DIVIDE_BY_ZERO: 'float_divide_by_zero', win32.EXCEPTION_FLT_INEXACT_RESULT: 'float_inexact_result', win32.EXCEPTION_FLT_INVALID_OPERATION: 'float_invalid_operation', win32.EXCEPTION_FLT_OVERFLOW: 'float_overflow', win32.EXCEPTION_FLT_STACK_CHECK: 'float_stack_check', win32.EXCEPTION_FLT_UNDERFLOW: 'float_underflow', win32.EXCEPTION_ILLEGAL_INSTRUCTION: 'illegal_instruction', win32.EXCEPTION_IN_PAGE_ERROR: 'in_page_error', win32.EXCEPTION_INT_DIVIDE_BY_ZERO: 'integer_divide_by_zero', win32.EXCEPTION_INT_OVERFLOW: 'integer_overflow', win32.EXCEPTION_INVALID_DISPOSITION: 'invalid_disposition', win32.EXCEPTION_NONCONTINUABLE_EXCEPTION: 'noncontinuable_exception', win32.EXCEPTION_PRIV_INSTRUCTION: 'privileged_instruction', win32.EXCEPTION_SINGLE_STEP: 'single_step', win32.EXCEPTION_STACK_OVERFLOW: 'stack_overflow', win32.EXCEPTION_GUARD_PAGE: 'guard_page', win32.EXCEPTION_INVALID_HANDLE: 'invalid_handle', win32.EXCEPTION_POSSIBLE_DEADLOCK: 'possible_deadlock', win32.EXCEPTION_WX86_BREAKPOINT: 'wow64_breakpoint', win32.CONTROL_C_EXIT: 'control_c_exit', win32.DBG_CONTROL_C: 'debug_control_c', win32.MS_VC_EXCEPTION: 'ms_vc_exception'}
    __exceptionName = {win32.EXCEPTION_ACCESS_VIOLATION: 'EXCEPTION_ACCESS_VIOLATION', win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED: 'EXCEPTION_ARRAY_BOUNDS_EXCEEDED', win32.EXCEPTION_BREAKPOINT: 'EXCEPTION_BREAKPOINT', win32.EXCEPTION_DATATYPE_MISALIGNMENT: 'EXCEPTION_DATATYPE_MISALIGNMENT', win32.EXCEPTION_FLT_DENORMAL_OPERAND: 'EXCEPTION_FLT_DENORMAL_OPERAND', win32.EXCEPTION_FLT_DIVIDE_BY_ZERO: 'EXCEPTION_FLT_DIVIDE_BY_ZERO', win32.EXCEPTION_FLT_INEXACT_RESULT: 'EXCEPTION_FLT_INEXACT_RESULT', win32.EXCEPTION_FLT_INVALID_OPERATION: 'EXCEPTION_FLT_INVALID_OPERATION', win32.EXCEPTION_FLT_OVERFLOW: 'EXCEPTION_FLT_OVERFLOW', win32.EXCEPTION_FLT_STACK_CHECK: 'EXCEPTION_FLT_STACK_CHECK', win32.EXCEPTION_FLT_UNDERFLOW: 'EXCEPTION_FLT_UNDERFLOW', win32.EXCEPTION_ILLEGAL_INSTRUCTION: 'EXCEPTION_ILLEGAL_INSTRUCTION', win32.EXCEPTION_IN_PAGE_ERROR: 'EXCEPTION_IN_PAGE_ERROR', win32.EXCEPTION_INT_DIVIDE_BY_ZERO: 'EXCEPTION_INT_DIVIDE_BY_ZERO', win32.EXCEPTION_INT_OVERFLOW: 'EXCEPTION_INT_OVERFLOW', win32.EXCEPTION_INVALID_DISPOSITION: 'EXCEPTION_INVALID_DISPOSITION', win32.EXCEPTION_NONCONTINUABLE_EXCEPTION: 'EXCEPTION_NONCONTINUABLE_EXCEPTION', win32.EXCEPTION_PRIV_INSTRUCTION: 'EXCEPTION_PRIV_INSTRUCTION', win32.EXCEPTION_SINGLE_STEP: 'EXCEPTION_SINGLE_STEP', win32.EXCEPTION_STACK_OVERFLOW: 'EXCEPTION_STACK_OVERFLOW', win32.EXCEPTION_GUARD_PAGE: 'EXCEPTION_GUARD_PAGE', win32.EXCEPTION_INVALID_HANDLE: 'EXCEPTION_INVALID_HANDLE', win32.EXCEPTION_POSSIBLE_DEADLOCK: 'EXCEPTION_POSSIBLE_DEADLOCK', win32.EXCEPTION_WX86_BREAKPOINT: 'EXCEPTION_WX86_BREAKPOINT', win32.CONTROL_C_EXIT: 'CONTROL_C_EXIT', win32.DBG_CONTROL_C: 'DBG_CONTROL_C', win32.MS_VC_EXCEPTION: 'MS_VC_EXCEPTION'}
    __exceptionDescription = {win32.EXCEPTION_ACCESS_VIOLATION: 'Access violation', win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED: 'Array bounds exceeded', win32.EXCEPTION_BREAKPOINT: 'Breakpoint', win32.EXCEPTION_DATATYPE_MISALIGNMENT: 'Datatype misalignment', win32.EXCEPTION_FLT_DENORMAL_OPERAND: 'Float denormal operand', win32.EXCEPTION_FLT_DIVIDE_BY_ZERO: 'Float divide by zero', win32.EXCEPTION_FLT_INEXACT_RESULT: 'Float inexact result', win32.EXCEPTION_FLT_INVALID_OPERATION: 'Float invalid operation', win32.EXCEPTION_FLT_OVERFLOW: 'Float overflow', win32.EXCEPTION_FLT_STACK_CHECK: 'Float stack check', win32.EXCEPTION_FLT_UNDERFLOW: 'Float underflow', win32.EXCEPTION_ILLEGAL_INSTRUCTION: 'Illegal instruction', win32.EXCEPTION_IN_PAGE_ERROR: 'In-page error', win32.EXCEPTION_INT_DIVIDE_BY_ZERO: 'Integer divide by zero', win32.EXCEPTION_INT_OVERFLOW: 'Integer overflow', win32.EXCEPTION_INVALID_DISPOSITION: 'Invalid disposition', win32.EXCEPTION_NONCONTINUABLE_EXCEPTION: 'Noncontinuable exception', win32.EXCEPTION_PRIV_INSTRUCTION: 'Privileged instruction', win32.EXCEPTION_SINGLE_STEP: 'Single step event', win32.EXCEPTION_STACK_OVERFLOW: 'Stack limits overflow', win32.EXCEPTION_GUARD_PAGE: 'Guard page hit', win32.EXCEPTION_INVALID_HANDLE: 'Invalid handle', win32.EXCEPTION_POSSIBLE_DEADLOCK: 'Possible deadlock', win32.EXCEPTION_WX86_BREAKPOINT: 'WOW64 breakpoint', win32.CONTROL_C_EXIT: 'Control-C exit', win32.DBG_CONTROL_C: 'Debug Control-C', win32.MS_VC_EXCEPTION: 'Microsoft Visual C++ exception'}

    @property
    def eventMethod(self):
        return self.__exceptionMethod.get(self.get_exception_code(), 'unknown_exception')

    def get_exception_name(self):
        """
        @rtype:  str
        @return: Name of the exception as defined by the Win32 API.
        """
        code = self.get_exception_code()
        unk = HexDump.integer(code)
        return self.__exceptionName.get(code, unk)

    def get_exception_description(self):
        """
        @rtype:  str
        @return: User-friendly name of the exception.
        """
        code = self.get_exception_code()
        description = self.__exceptionDescription.get(code, None)
        if description is None:
            try:
                description = 'Exception code %s (%s)'
                description = description % (HexDump.integer(code), ctypes.FormatError(code))
            except OverflowError:
                description = 'Exception code %s' % HexDump.integer(code)
        return description

    def is_first_chance(self):
        """
        @rtype:  bool
        @return: C{True} for first chance exceptions, C{False} for last chance.
        """
        return self.raw.u.Exception.dwFirstChance != 0

    def is_last_chance(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_first_chance}.
        """
        return not self.is_first_chance()

    def is_noncontinuable(self):
        """
        @see: U{http://msdn.microsoft.com/en-us/library/aa363082(VS.85).aspx}

        @rtype:  bool
        @return: C{True} if the exception is noncontinuable,
            C{False} otherwise.

            Attempting to continue a noncontinuable exception results in an
            EXCEPTION_NONCONTINUABLE_EXCEPTION exception to be raised.
        """
        return bool(self.raw.u.Exception.ExceptionRecord.ExceptionFlags & win32.EXCEPTION_NONCONTINUABLE)

    def is_continuable(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_noncontinuable}.
        """
        return not self.is_noncontinuable()

    def is_user_defined_exception(self):
        """
        Determines if this is an user-defined exception. User-defined
        exceptions may contain any exception code that is not system reserved.

        Often the exception code is also a valid Win32 error code, but that's
        up to the debugged application.

        @rtype:  bool
        @return: C{True} if the exception is user-defined, C{False} otherwise.
        """
        return self.get_exception_code() & 268435456 == 0

    def is_system_defined_exception(self):
        """
        @rtype:  bool
        @return: The opposite of L{is_user_defined_exception}.
        """
        return not self.is_user_defined_exception()

    def get_exception_code(self):
        """
        @rtype:  int
        @return: Exception code as defined by the Win32 API.
        """
        return self.raw.u.Exception.ExceptionRecord.ExceptionCode

    def get_exception_address(self):
        """
        @rtype:  int
        @return: Memory address where the exception occured.
        """
        address = self.raw.u.Exception.ExceptionRecord.ExceptionAddress
        if address is None:
            address = 0
        return address

    def get_exception_information(self, index):
        """
        @type  index: int
        @param index: Index into the exception information block.

        @rtype:  int
        @return: Exception information DWORD.
        """
        if index < 0 or index > win32.EXCEPTION_MAXIMUM_PARAMETERS:
            raise IndexError('Array index out of range: %s' % repr(index))
        info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
        value = info[index]
        if value is None:
            value = 0
        return value

    def get_exception_information_as_list(self):
        """
        @rtype:  list( int )
        @return: Exception information block.
        """
        info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
        data = list()
        for index in compat.xrange(0, win32.EXCEPTION_MAXIMUM_PARAMETERS):
            value = info[index]
            if value is None:
                value = 0
            data.append(value)
        return data

    def get_fault_type(self):
        """
        @rtype:  int
        @return: Access violation type.
            Should be one of the following constants:

             - L{win32.EXCEPTION_READ_FAULT}
             - L{win32.EXCEPTION_WRITE_FAULT}
             - L{win32.EXCEPTION_EXECUTE_FAULT}

        @note: This method is only meaningful for access violation exceptions,
            in-page memory error exceptions and guard page exceptions.

        @raise NotImplementedError: Wrong kind of exception.
        """
        if self.get_exception_code() not in (win32.EXCEPTION_ACCESS_VIOLATION, win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_GUARD_PAGE):
            msg = 'This method is not meaningful for %s.'
            raise NotImplementedError(msg % self.get_exception_name())
        return self.get_exception_information(0)

    def get_fault_address(self):
        """
        @rtype:  int
        @return: Access violation memory address.

        @note: This method is only meaningful for access violation exceptions,
            in-page memory error exceptions and guard page exceptions.

        @raise NotImplementedError: Wrong kind of exception.
        """
        if self.get_exception_code() not in (win32.EXCEPTION_ACCESS_VIOLATION, win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_GUARD_PAGE):
            msg = 'This method is not meaningful for %s.'
            raise NotImplementedError(msg % self.get_exception_name())
        return self.get_exception_information(1)

    def get_ntstatus_code(self):
        """
        @rtype:  int
        @return: NTSTATUS status code that caused the exception.

        @note: This method is only meaningful for in-page memory error
            exceptions.

        @raise NotImplementedError: Not an in-page memory error.
        """
        if self.get_exception_code() != win32.EXCEPTION_IN_PAGE_ERROR:
            msg = 'This method is only meaningful for in-page memory error exceptions.'
            raise NotImplementedError(msg)
        return self.get_exception_information(2)

    def is_nested(self):
        """
        @rtype:  bool
        @return: Returns C{True} if there are additional exception records
            associated with this exception. This would mean the exception
            is nested, that is, it was triggered while trying to handle
            at least one previous exception.
        """
        return bool(self.raw.u.Exception.ExceptionRecord.ExceptionRecord)

    def get_raw_exception_record_list(self):
        """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{win32.EXCEPTION_RECORD} )
        @return:
            List of raw exception record structures as used by the Win32 API.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
        nested = list()
        record = self.raw.u.Exception
        while True:
            record = record.ExceptionRecord
            if not record:
                break
            nested.append(record)
        return nested

    def get_nested_exceptions(self):
        """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{ExceptionEvent} )
        @return:
            List of ExceptionEvent objects representing each exception record
            found in this event.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
        nested = [self]
        raw = self.raw
        dwDebugEventCode = raw.dwDebugEventCode
        dwProcessId = raw.dwProcessId
        dwThreadId = raw.dwThreadId
        dwFirstChance = raw.u.Exception.dwFirstChance
        record = raw.u.Exception.ExceptionRecord
        while True:
            record = record.ExceptionRecord
            if not record:
                break
            raw = win32.DEBUG_EVENT()
            raw.dwDebugEventCode = dwDebugEventCode
            raw.dwProcessId = dwProcessId
            raw.dwThreadId = dwThreadId
            raw.u.Exception.ExceptionRecord = record
            raw.u.Exception.dwFirstChance = dwFirstChance
            event = EventFactory.get(self.debug, raw)
            nested.append(event)
        return nested