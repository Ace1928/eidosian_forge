import sys  # @NoMove
import os
from _pydevd_bundle import pydevd_constants
import atexit
import dis
import io
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import itertools
import traceback
import weakref
import getpass as getpass_mod
import functools
import pydevd_file_utils
from _pydev_bundle import pydev_imports, pydev_log
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydev_bundle._pydev_saved_modules import threading, time, thread
from _pydevd_bundle import pydevd_extension_utils, pydevd_frame_utils
from _pydevd_bundle.pydevd_filtering import FilesFiltering, glob_matches_path
from _pydevd_bundle import pydevd_io, pydevd_vm_type, pydevd_defaults
from _pydevd_bundle import pydevd_utils
from _pydevd_bundle import pydevd_runpy
from _pydev_bundle.pydev_console_utils import DebugConsoleStdIn
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import ExceptionBreakpoint, get_exception_breakpoint
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, CMD_STEP_INTO, CMD_SET_BREAK,
from _pydevd_bundle.pydevd_constants import (get_thread_id, get_current_thread_id,
from _pydevd_bundle.pydevd_defaults import PydevdCustomization  # Note: import alias used on pydev_monkey.
from _pydevd_bundle.pydevd_custom_frames import CustomFramesContainer, custom_frames_container_init
from _pydevd_bundle.pydevd_dont_trace_files import DONT_TRACE, PYDEV_FILE, LIB_FILE, DONT_TRACE_DIRS
from _pydevd_bundle.pydevd_extension_api import DebuggerEventHandler
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, remove_exception_from_frame
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_trace_dispatch import (
from _pydevd_bundle.pydevd_utils import save_main_module, is_current_thread_main_thread, \
from _pydevd_frame_eval.pydevd_frame_eval_main import (
import pydev_ipython  # @UnusedImport
from _pydevd_bundle.pydevd_source_mapping import SourceMapping
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import ThreadingLogger, AsyncioLogger, send_concurrency_message, cur_time
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import wrap_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from pydevd_file_utils import get_fullname, get_package_dir
from os.path import abspath as os_path_abspath
import pydevd_tracing
from _pydevd_bundle.pydevd_comm import (InternalThreadCommand, InternalThreadCommandForAnyThread,
from _pydevd_bundle.pydevd_comm import(InternalConsoleExec,
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread, mark_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_process_net_command_json import PyDevJsonCommandProcessor
from _pydevd_bundle.pydevd_process_net_command import process_net_command
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info, collect_return_info, collect_try_except_info_from_source
from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
from socket import SHUT_RDWR
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_timeout import TimeoutTracker
from _pydevd_bundle.pydevd_thread_lifecycle import suspend_all_threads, mark_thread_suspended
from _pydevd_bundle.pydevd_plugin_utils import PluginManager
def _locked_settrace(host, stdout_to_server, stderr_to_server, port, suspend, trace_only_current_thread, patch_multiprocessing, stop_at_frame, block_until_connected, wait_for_ready_to_run, dont_trace_start_patterns, dont_trace_end_patterns, access_token, client_access_token, __setup_holder__, notify_stdin):
    if patch_multiprocessing:
        try:
            from _pydev_bundle import pydev_monkey
        except:
            pass
        else:
            pydev_monkey.patch_new_process_functions()
    if host is None:
        from _pydev_bundle import pydev_localhost
        host = pydev_localhost.get_localhost()
    global _global_redirect_stdout_to_server
    global _global_redirect_stderr_to_server
    py_db = get_global_debugger()
    if __setup_holder__:
        SetupHolder.setup = __setup_holder__
    if py_db is None:
        py_db = PyDB()
        pydevd_vm_type.setup_type()
        if SetupHolder.setup is None:
            setup = {'client': host, 'server': False, 'port': int(port), 'multiprocess': patch_multiprocessing, 'skip-notify-stdin': not notify_stdin}
            SetupHolder.setup = setup
        if access_token is not None:
            py_db.authentication.access_token = access_token
            SetupHolder.setup['access-token'] = access_token
        if client_access_token is not None:
            py_db.authentication.client_access_token = client_access_token
            SetupHolder.setup['client-access-token'] = client_access_token
        if block_until_connected:
            py_db.connect(host, port)
        else:
            py_db.writer = WriterThread(NULL, py_db, terminate_on_socket_close=False)
            py_db.create_wait_for_connection_thread()
        if dont_trace_start_patterns or dont_trace_end_patterns:
            PyDevdAPI().set_dont_trace_start_end_patterns(py_db, dont_trace_start_patterns, dont_trace_end_patterns)
        _global_redirect_stdout_to_server = stdout_to_server
        _global_redirect_stderr_to_server = stderr_to_server
        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()
        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()
        if notify_stdin:
            patch_stdin()
        t = threadingCurrentThread()
        additional_info = set_additional_thread_info(t)
        if not wait_for_ready_to_run:
            py_db.ready_to_run = True
        py_db.wait_for_ready_to_run()
        py_db.start_auxiliary_daemon_threads()
        try:
            if INTERACTIVE_MODE_AVAILABLE:
                py_db.init_gui_support()
        except:
            pydev_log.exception('Matplotlib support in debugger failed')
        if trace_only_current_thread:
            py_db.enable_tracing()
        else:
            py_db.patch_threads()
            py_db.enable_tracing(py_db.trace_dispatch, apply_to_all_threads=True)
            py_db.set_tracing_for_untraced_contexts()
        py_db.set_trace_for_frame_and_parents(get_frame().f_back)
        with CustomFramesContainer.custom_frames_lock:
            for _frameId, custom_frame in CustomFramesContainer.custom_frames.items():
                py_db.set_trace_for_frame_and_parents(custom_frame.frame)
    else:
        if access_token is not None:
            py_db.authentication.access_token = access_token
        if client_access_token is not None:
            py_db.authentication.client_access_token = client_access_token
        py_db.set_trace_for_frame_and_parents(get_frame().f_back)
        t = threadingCurrentThread()
        additional_info = set_additional_thread_info(t)
        if trace_only_current_thread:
            py_db.enable_tracing()
        else:
            py_db.patch_threads()
            py_db.enable_tracing(py_db.trace_dispatch, apply_to_all_threads=True)
    if suspend:
        if stop_at_frame is not None:
            additional_info.pydev_state = STATE_RUN
            additional_info.pydev_original_step_cmd = CMD_STEP_OVER
            additional_info.pydev_step_cmd = CMD_STEP_OVER
            additional_info.pydev_step_stop = stop_at_frame
            additional_info.suspend_type = PYTHON_SUSPEND
        else:
            py_db.set_suspend(t, CMD_SET_BREAK)