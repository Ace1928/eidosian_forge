import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
class IOUtils(object):
    """Asyncronous IO helper class."""

    def __init__(self):
        self._win32_utils = win32utils.Win32Utils()

    def _run_and_check_output(self, *args, **kwargs):
        eventlet_blocking_mode = kwargs.get('eventlet_nonblocking_mode', False)
        kwargs.update(kernel32_lib_func=True, failure_exc=exceptions.Win32IOException, eventlet_nonblocking_mode=eventlet_blocking_mode)
        return self._win32_utils.run_and_check_output(*args, **kwargs)

    def create_pipe(self, security_attributes=None, size=0, inherit_handle=False):
        """Create an anonymous pipe.

        The main advantage of this method over os.pipe is that it allows
        creating inheritable pipe handles (which is flawed on most Python
        versions).
        """
        r = wintypes.HANDLE()
        w = wintypes.HANDLE()
        if inherit_handle and (not security_attributes):
            security_attributes = wintypes.SECURITY_ATTRIBUTES()
            security_attributes.bInheritHandle = inherit_handle
            security_attributes.nLength = ctypes.sizeof(security_attributes)
        self._run_and_check_output(kernel32.CreatePipe, ctypes.byref(r), ctypes.byref(w), ctypes.byref(security_attributes) if security_attributes else None, size)
        return (r.value, w.value)

    @_utils.retry_decorator(exceptions=exceptions.Win32IOException, max_sleep_time=2)
    def wait_named_pipe(self, pipe_name, timeout=WAIT_PIPE_DEFAULT_TIMEOUT):
        """Wait a given amount of time for a pipe to become available."""
        self._run_and_check_output(kernel32.WaitNamedPipeW, ctypes.c_wchar_p(pipe_name), timeout * units.k)

    def open(self, path, desired_access=0, share_mode=0, creation_disposition=0, flags_and_attributes=0):
        error_ret_vals = [w_const.INVALID_HANDLE_VALUE]
        handle = self._run_and_check_output(kernel32.CreateFileW, ctypes.c_wchar_p(path), desired_access, share_mode, None, creation_disposition, flags_and_attributes, None, error_ret_vals=error_ret_vals)
        return handle

    def close_handle(self, handle):
        self._run_and_check_output(kernel32.CloseHandle, handle)

    def cancel_io(self, handle, overlapped_structure=None, ignore_invalid_handle=False):
        """Cancels pending IO on specified handle.

        If an overlapped structure is passed, only the IO requests that
        were issued with the specified overlapped structure are canceled.
        """
        ignored_error_codes = [w_const.ERROR_NOT_FOUND]
        if ignore_invalid_handle:
            ignored_error_codes.append(w_const.ERROR_INVALID_HANDLE)
        lp_overlapped = ctypes.byref(overlapped_structure) if overlapped_structure else None
        self._run_and_check_output(kernel32.CancelIoEx, handle, lp_overlapped, ignored_error_codes=ignored_error_codes)

    def _wait_io_completion(self, event):
        self._run_and_check_output(kernel32.WaitForSingleObjectEx, event, WAIT_INFINITE_TIMEOUT, True, error_ret_vals=[w_const.WAIT_FAILED])

    def set_event(self, event):
        self._run_and_check_output(kernel32.SetEvent, event)

    def _reset_event(self, event):
        self._run_and_check_output(kernel32.ResetEvent, event)

    def _create_event(self, event_attributes=None, manual_reset=True, initial_state=False, name=None):
        return self._run_and_check_output(kernel32.CreateEventW, event_attributes, manual_reset, initial_state, name, error_ret_vals=[None])

    def get_completion_routine(self, callback=None):

        def _completion_routine(error_code, num_bytes, lpOverLapped):
            """Sets the completion event and executes callback, if passed."""
            overlapped = ctypes.cast(lpOverLapped, wintypes.LPOVERLAPPED).contents
            self.set_event(overlapped.hEvent)
            if callback:
                callback(num_bytes)
        return wintypes.LPOVERLAPPED_COMPLETION_ROUTINE(_completion_routine)

    def get_new_overlapped_structure(self):
        """Structure used for asynchronous IO operations."""
        hEvent = self._create_event()
        overlapped_structure = wintypes.OVERLAPPED()
        overlapped_structure.hEvent = hEvent
        return overlapped_structure

    def read(self, handle, buff, num_bytes, overlapped_structure, completion_routine):
        self._reset_event(overlapped_structure.hEvent)
        self._run_and_check_output(kernel32.ReadFileEx, handle, buff, num_bytes, ctypes.byref(overlapped_structure), completion_routine)
        self._wait_io_completion(overlapped_structure.hEvent)

    def read_file(self, handle, buff, num_bytes, overlapped_structure=None):
        num_bytes_read = wintypes.DWORD(0)
        overlapped_structure_ref = ctypes.byref(overlapped_structure) if overlapped_structure else None
        self._run_and_check_output(kernel32.ReadFile, handle, buff, num_bytes, ctypes.byref(num_bytes_read), overlapped_structure_ref)
        return num_bytes_read.value

    def write(self, handle, buff, num_bytes, overlapped_structure, completion_routine):
        self._reset_event(overlapped_structure.hEvent)
        self._run_and_check_output(kernel32.WriteFileEx, handle, buff, num_bytes, ctypes.byref(overlapped_structure), completion_routine)
        self._wait_io_completion(overlapped_structure.hEvent)

    def write_file(self, handle, buff, num_bytes, overlapped_structure=None):
        num_bytes_written = wintypes.DWORD(0)
        overlapped_structure_ref = ctypes.byref(overlapped_structure) if overlapped_structure else None
        self._run_and_check_output(kernel32.WriteFile, handle, buff, num_bytes, ctypes.byref(num_bytes_written), overlapped_structure_ref)
        return num_bytes_written.value

    @classmethod
    def get_buffer(cls, buff_size, data=None):
        buff = (ctypes.c_ubyte * buff_size)()
        if data:
            cls.write_buffer_data(buff, data)
        return buff

    @staticmethod
    def get_buffer_data(buff, num_bytes):
        return bytes(bytearray(buff[:num_bytes]))

    @staticmethod
    def write_buffer_data(buff, data):
        for i, c in enumerate(data):
            buff[i] = struct.unpack('B', six.b(c))[0]