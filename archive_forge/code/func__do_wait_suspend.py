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
def _do_wait_suspend(self, thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker):
    info = thread.additional_info
    info.step_in_initial_location = None
    keep_suspended = False
    with self._main_lock:
        activate_gui = info.pydev_state == STATE_SUSPEND and (not self.pydb_disposed)
    in_main_thread = is_current_thread_main_thread()
    if activate_gui and in_main_thread:
        self._activate_gui_if_needed()
    while True:
        with self._main_lock:
            if info.pydev_state != STATE_SUSPEND or (self.pydb_disposed and (not self.terminate_requested)):
                break
        if in_main_thread and self.gui_in_use:
            self._call_input_hook()
        self.process_internal_commands()
        time.sleep(0.01)
    self.cancel_async_evaluation(get_current_thread_id(thread), str(id(frame)))
    if info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
        info.step_in_initial_location = (frame, frame.f_lineno)
        if frame.f_code.co_flags & 128:
            info.pydev_step_cmd = CMD_STEP_INTO_COROUTINE
            info.pydev_step_stop = frame
            self.set_trace_for_frame_and_parents(frame)
        else:
            info.pydev_step_stop = None
            self.set_trace_for_frame_and_parents(frame)
    elif info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO):
        info.pydev_step_stop = frame
        self.set_trace_for_frame_and_parents(frame)
    elif info.pydev_step_cmd == CMD_RUN_TO_LINE or info.pydev_step_cmd == CMD_SET_NEXT_STATEMENT:
        info.pydev_step_stop = None
        self.set_trace_for_frame_and_parents(frame)
        stop = False
        response_msg = ''
        try:
            stop, _old_line, response_msg = self.set_next_statement(frame, event, info.pydev_func_name, info.pydev_next_line)
        except ValueError as e:
            response_msg = '%s' % e
        finally:
            seq = info.pydev_message
            cmd = self.cmd_factory.make_set_next_stmnt_status_message(seq, stop, response_msg)
            self.writer.add_command(cmd)
            info.pydev_message = ''
        if stop:
            frames_tracker.untrack_all()
            cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
            self.writer.add_command(cmd)
            info.pydev_state = STATE_SUSPEND
            thread.stop_reason = CMD_SET_NEXT_STATEMENT
            keep_suspended = True
        else:
            info.pydev_original_step_cmd = -1
            info.pydev_step_cmd = -1
            info.pydev_state = STATE_SUSPEND
            thread.stop_reason = CMD_THREAD_SUSPEND
            return self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)
    elif info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
        back_frame = frame.f_back
        force_check_project_scope = info.pydev_step_cmd == CMD_STEP_RETURN_MY_CODE
        if force_check_project_scope or self.is_files_filter_enabled:
            while back_frame is not None:
                if self.apply_files_filter(back_frame, back_frame.f_code.co_filename, force_check_project_scope):
                    frame = back_frame
                    back_frame = back_frame.f_back
                else:
                    break
        if back_frame is not None:
            info.pydev_step_stop = frame
            self.set_trace_for_frame_and_parents(frame)
        else:
            info.pydev_step_stop = None
            info.pydev_original_step_cmd = -1
            info.pydev_step_cmd = -1
            info.pydev_state = STATE_RUN
    if PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING:
        info.pydev_use_scoped_step_frame = False
        if info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
            f = frame.f_back
            if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                f = f.f_back
                if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                    info.pydev_use_scoped_step_frame = True
                    pydev_log.info('Using (ipython) scoped stepping.')
            del f
    del frame
    cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
    self.writer.add_command(cmd)
    with CustomFramesContainer.custom_frames_lock:
        for frame_id in from_this_thread:
            self.writer.add_command(self.cmd_factory.make_thread_killed_message(frame_id))
    return keep_suspended