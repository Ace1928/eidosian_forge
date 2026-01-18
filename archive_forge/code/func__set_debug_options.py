import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
def _set_debug_options(self, py_db, args, start_reason):
    rules = args.get('rules')
    stepping_resumes_all_threads = args.get('steppingResumesAllThreads', True)
    self.api.set_stepping_resumes_all_threads(py_db, stepping_resumes_all_threads)
    terminate_child_processes = args.get('terminateChildProcesses', True)
    self.api.set_terminate_child_processes(py_db, terminate_child_processes)
    terminate_keyboard_interrupt = args.get('onTerminate', 'kill') == 'KeyboardInterrupt'
    self.api.set_terminate_keyboard_interrupt(py_db, terminate_keyboard_interrupt)
    variable_presentation = args.get('variablePresentation', None)
    if isinstance(variable_presentation, dict):

        def get_variable_presentation(setting, default):
            value = variable_presentation.get(setting, default)
            if value not in ('group', 'inline', 'hide'):
                pydev_log.info('The value set for "%s" (%s) in the variablePresentation is not valid. Valid values are: "group", "inline", "hide"' % (setting, value))
                value = default
            return value
        default = get_variable_presentation('all', 'group')
        special_presentation = get_variable_presentation('special', default)
        function_presentation = get_variable_presentation('function', default)
        class_presentation = get_variable_presentation('class', default)
        protected_presentation = get_variable_presentation('protected', default)
        self.api.set_variable_presentation(py_db, self.api.VariablePresentation(special_presentation, function_presentation, class_presentation, protected_presentation))
    exclude_filters = []
    if rules is not None:
        exclude_filters = _convert_rules_to_exclude_filters(rules, lambda msg: self.api.send_error_message(py_db, msg))
    self.api.set_exclude_filters(py_db, exclude_filters)
    debug_options = _extract_debug_options(args.get('options'), args.get('debugOptions'))
    self._options.update_fom_debug_options(debug_options)
    self._options.update_from_args(args)
    self.api.set_use_libraries_filter(py_db, self._options.just_my_code)
    if self._options.client_os:
        self.api.set_ide_os(self._options.client_os)
    path_mappings = []
    for pathMapping in args.get('pathMappings', []):
        localRoot = pathMapping.get('localRoot', '')
        remoteRoot = pathMapping.get('remoteRoot', '')
        remoteRoot = self._resolve_remote_root(localRoot, remoteRoot)
        if localRoot != '' and remoteRoot != '':
            path_mappings.append((localRoot, remoteRoot))
    if bool(path_mappings):
        pydevd_file_utils.setup_client_server_paths(path_mappings)
    resolve_symlinks = args.get('resolveSymlinks', None)
    if resolve_symlinks is not None:
        pydevd_file_utils.set_resolve_symlinks(resolve_symlinks)
    redirecting = args.get('isOutputRedirected')
    if self._options.redirect_output:
        py_db.enable_output_redirection(True, True)
        redirecting = True
    else:
        py_db.enable_output_redirection(False, False)
    py_db.is_output_redirected = redirecting
    self.api.set_show_return_values(py_db, self._options.show_return_value)
    if not self._options.break_system_exit_zero:
        ignore_system_exit_codes = [0, None]
        if self._options.django_debug or self._options.flask_debug:
            ignore_system_exit_codes += [3]
        self.api.set_ignore_system_exit_codes(py_db, ignore_system_exit_codes)
    auto_reload = args.get('autoReload', {})
    if not isinstance(auto_reload, dict):
        pydev_log.info('Expected autoReload to be a dict. Received: %s' % (auto_reload,))
        auto_reload = {}
    enable_auto_reload = auto_reload.get('enable', False)
    watch_dirs = auto_reload.get('watchDirectories')
    if not watch_dirs:
        watch_dirs = []
        program = args.get('program')
        if program:
            if os.path.isdir(program):
                watch_dirs.append(program)
            else:
                watch_dirs.append(os.path.dirname(program))
        watch_dirs.append(os.path.abspath('.'))
        argv = getattr(sys, 'argv', [])
        if argv:
            f = argv[0]
            if f:
                if os.path.isdir(f):
                    watch_dirs.append(f)
                else:
                    watch_dirs.append(os.path.dirname(f))
    if not isinstance(watch_dirs, (list, set, tuple)):
        watch_dirs = (watch_dirs,)
    new_watch_dirs = set()
    for w in watch_dirs:
        try:
            new_watch_dirs.add(pydevd_file_utils.get_path_with_real_case(pydevd_file_utils.absolute_path(w)))
        except Exception:
            pydev_log.exception('Error adding watch dir: %s', w)
    watch_dirs = new_watch_dirs
    poll_target_time = auto_reload.get('pollingInterval', 1)
    exclude_patterns = auto_reload.get('exclude', ('**/.git/**', '**/__pycache__/**', '**/node_modules/**', '**/.metadata/**', '**/site-packages/**'))
    include_patterns = auto_reload.get('include', ('**/*.py', '**/*.pyw'))
    self.api.setup_auto_reload_watcher(py_db, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns)
    if self._options.stop_on_entry and start_reason == 'launch':
        self.api.stop_on_entry()
    self.api.set_gui_event_loop(py_db, self._options.gui_event_loop)