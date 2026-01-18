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
class AbstractSingleNotificationBehavior(object):
    """
    The basic usage should be:

    # Increment the request time for the suspend.
    single_notification_behavior.increment_suspend_time()

    # Notify that this is a pause request (when a pause, not a breakpoint).
    single_notification_behavior.on_pause()

    # Mark threads to be suspended.
    set_suspend(...)

    # On do_wait_suspend, use notify_thread_suspended:
    def do_wait_suspend(...):
        with single_notification_behavior.notify_thread_suspended(thread_id, thread, reason):
            ...
    """
    __slots__ = ['_last_resume_notification_time', '_last_suspend_notification_time', '_lock', '_next_request_time', '_suspend_time_request', '_suspended_thread_id_to_thread', '_pause_requested', '_py_db']
    NOTIFY_OF_PAUSE_TIMEOUT = 0.5

    def __init__(self, py_db):
        self._py_db = weakref.ref(py_db)
        self._next_request_time = partial(next, itertools.count())
        self._last_suspend_notification_time = -1
        self._last_resume_notification_time = -1
        self._suspend_time_request = self._next_request_time()
        self._lock = thread.allocate_lock()
        self._suspended_thread_id_to_thread = {}
        self._pause_requested = False

    def send_suspend_notification(self, thread_id, thread, stop_reason):
        raise AssertionError('abstract: subclasses must override.')

    def send_resume_notification(self, thread_id):
        raise AssertionError('abstract: subclasses must override.')

    def increment_suspend_time(self):
        with self._lock:
            self._suspend_time_request = self._next_request_time()

    def on_pause(self):
        with self._lock:
            self._pause_requested = True
            global_suspend_time = self._suspend_time_request
        py_db = self._py_db()
        if py_db is not None:
            py_db.timeout_tracker.call_on_timeout(self.NOTIFY_OF_PAUSE_TIMEOUT, self._notify_after_timeout, kwargs={'global_suspend_time': global_suspend_time})

    def _notify_after_timeout(self, global_suspend_time):
        with self._lock:
            if self._suspended_thread_id_to_thread:
                if global_suspend_time > self._last_suspend_notification_time:
                    self._last_suspend_notification_time = global_suspend_time
                    pydev_log.info('Sending suspend notification after timeout.')
                    thread_id, thread = next(iter(self._suspended_thread_id_to_thread.items()))
                    self.send_suspend_notification(thread_id, thread, CMD_THREAD_SUSPEND)

    def on_thread_suspend(self, thread_id, thread, stop_reason):
        with self._lock:
            pause_requested = self._pause_requested
            if pause_requested:
                self._pause_requested = False
            self._suspended_thread_id_to_thread[thread_id] = thread
            if stop_reason != CMD_THREAD_SUSPEND or pause_requested:
                if self._suspend_time_request > self._last_suspend_notification_time:
                    pydev_log.info('Sending suspend notification.')
                    self._last_suspend_notification_time = self._suspend_time_request
                    self.send_suspend_notification(thread_id, thread, stop_reason)
                else:
                    pydev_log.info('Suspend not sent (it was already sent). Last suspend % <= Last resume %s', self._last_suspend_notification_time, self._last_resume_notification_time)
            else:
                pydev_log.info('Suspend not sent because stop reason is thread suspend and pause was not requested.')

    def on_thread_resume(self, thread_id, thread):
        with self._lock:
            self._suspended_thread_id_to_thread.pop(thread_id)
            if self._last_resume_notification_time < self._last_suspend_notification_time:
                pydev_log.info('Sending resume notification.')
                self._last_resume_notification_time = self._last_suspend_notification_time
                self.send_resume_notification(thread_id)
            else:
                pydev_log.info('Resume not sent (it was already sent). Last resume %s >= Last suspend %s', self._last_resume_notification_time, self._last_suspend_notification_time)

    @contextmanager
    def notify_thread_suspended(self, thread_id, thread, stop_reason):
        self.on_thread_suspend(thread_id, thread, stop_reason)
        try:
            yield
        finally:
            self.on_thread_resume(thread_id, thread)