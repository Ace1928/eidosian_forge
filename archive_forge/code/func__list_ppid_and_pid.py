import sys
import bisect
import types
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize
def _list_ppid_and_pid():
    _TH32CS_SNAPPROCESS = 2

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [('dwSize', ctypes.c_uint32), ('cntUsage', ctypes.c_uint32), ('th32ProcessID', ctypes.c_uint32), ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', ctypes.c_uint32), ('cntThreads', ctypes.c_uint32), ('th32ParentProcessID', ctypes.c_uint32), ('pcPriClassBase', ctypes.c_long), ('dwFlags', ctypes.c_uint32), ('szExeFile', ctypes.c_char * 260)]
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    ppid_and_pids = []
    try:
        process_entry = PROCESSENTRY32()
        process_entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32First(ctypes.c_void_p(snapshot), ctypes.byref(process_entry)):
            pydev_log.critical('Process32First failed (getting process from CreateToolhelp32Snapshot).')
        else:
            while True:
                ppid_and_pids.append((process_entry.th32ParentProcessID, process_entry.th32ProcessID))
                if not kernel32.Process32Next(ctypes.c_void_p(snapshot), ctypes.byref(process_entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)
    return ppid_and_pids