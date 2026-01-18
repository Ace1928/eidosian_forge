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
def add_breakpoint(self, py_db, original_filename, breakpoint_type, breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition, is_logpoint, adjust_line=False, on_changed_breakpoint_state=None):
    """
        :param str original_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function and its final value will be available in the returned _AddBreakpointResult.

        :param str breakpoint_type:
            One of: 'python-line', 'django-line', 'jinja2-line'.

        :param int breakpoint_id:

        :param int line:
            Note: it's possible that a new line was actually used. If that's the case its
            final value will be available in the returned _AddBreakpointResult.

        :param condition:
            Either None or the condition to activate the breakpoint.

        :param str func_name:
            If "None" (str), may hit in any context.
            Empty string will hit only top level.
            Any other value must match the scope of the method to be matched.

        :param str expression:
            None or the expression to be evaluated.

        :param suspend_policy:
            Either "NONE" (to suspend only the current thread when the breakpoint is hit) or
            "ALL" (to suspend all threads when a breakpoint is hit).

        :param str hit_condition:
            An expression where `@HIT@` will be replaced by the number of hits.
            i.e.: `@HIT@ == x` or `@HIT@ >= x`

        :param bool is_logpoint:
            If True and an expression is passed, pydevd will create an io message command with the
            result of the evaluation.

        :param bool adjust_line:
            If True, the breakpoint line should be adjusted if the current line doesn't really
            match an executable line (if possible).

        :param callable on_changed_breakpoint_state:
            This is called when something changed internally on the breakpoint after it was initially
            added (for instance, template file_to_line_to_breakpoints could be signaled as invalid initially and later
            when the related template is loaded, if the line is valid it could be marked as valid).

            The signature for the callback should be:
                on_changed_breakpoint_state(breakpoint_id: int, add_breakpoint_result: _AddBreakpointResult)

                Note that the add_breakpoint_result should not be modified by the callback (the
                implementation may internally reuse the same instance multiple times).

        :return _AddBreakpointResult:
        """
    assert original_filename.__class__ == str, 'Expected str, found: %s' % (original_filename.__class__,)
    original_filename_normalized = pydevd_file_utils.normcase_from_client(original_filename)
    pydev_log.debug('Request for breakpoint in: %s line: %s', original_filename, line)
    original_line = line
    api_add_breakpoint_params = (original_filename, breakpoint_type, breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition, is_logpoint)
    translated_filename = self.filename_to_server(original_filename)
    pydev_log.debug('Breakpoint (after path translation) in: %s line: %s', translated_filename, line)
    func_name = self.to_str(func_name)
    assert translated_filename.__class__ == str
    assert func_name.__class__ == str
    source_mapped_filename, new_line, multi_mapping_applied = py_db.source_mapping.map_to_server(translated_filename, line)
    if multi_mapping_applied:
        pydev_log.debug('Breakpoint (after source mapping) in: %s line: %s', source_mapped_filename, new_line)
        result = self._AddBreakpointResult(breakpoint_id, original_filename, line, original_line)
        translated_absolute_filename = source_mapped_filename
        canonical_normalized_filename = pydevd_file_utils.normcase(source_mapped_filename)
        line = new_line
    else:
        translated_absolute_filename = pydevd_file_utils.absolute_path(translated_filename)
        canonical_normalized_filename = pydevd_file_utils.canonical_normalized_path(translated_filename)
        if adjust_line and (not translated_absolute_filename.startswith('<')):
            try:
                lines = sorted(_get_code_lines(translated_absolute_filename))
            except Exception:
                pass
            else:
                if line not in lines:
                    idx = bisect.bisect_left(lines, line)
                    if idx > 0:
                        line = lines[idx - 1]
        result = self._AddBreakpointResult(breakpoint_id, original_filename, line, original_line)
    py_db.api_received_breakpoints[original_filename_normalized, breakpoint_id] = (canonical_normalized_filename, api_add_breakpoint_params)
    if not translated_absolute_filename.startswith('<'):
        if not pydevd_file_utils.exists(translated_absolute_filename):
            result.error_code = self.ADD_BREAKPOINT_FILE_NOT_FOUND
            return result
        if py_db.is_files_filter_enabled and (not py_db.get_require_module_for_filters()) and py_db.apply_files_filter(self._DummyFrame(translated_absolute_filename), translated_absolute_filename, False):
            result.error_code = self.ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS
    if breakpoint_type == 'python-line':
        added_breakpoint = LineBreakpoint(breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition=hit_condition, is_logpoint=is_logpoint)
        file_to_line_to_breakpoints = py_db.breakpoints
        file_to_id_to_breakpoint = py_db.file_to_id_to_line_breakpoint
        supported_type = True
    else:
        add_plugin_breakpoint_result = None
        plugin = py_db.get_plugin_lazy_init()
        if plugin is not None:
            add_plugin_breakpoint_result = plugin.add_breakpoint('add_line_breakpoint', py_db, breakpoint_type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition=hit_condition, is_logpoint=is_logpoint, add_breakpoint_result=result, on_changed_breakpoint_state=on_changed_breakpoint_state)
        if add_plugin_breakpoint_result is not None:
            supported_type = True
            added_breakpoint, file_to_line_to_breakpoints = add_plugin_breakpoint_result
            file_to_id_to_breakpoint = py_db.file_to_id_to_plugin_breakpoint
        else:
            supported_type = False
    if not supported_type:
        raise NameError(breakpoint_type)
    pydev_log.debug('Added breakpoint:%s - line:%s - func_name:%s\n', canonical_normalized_filename, line, func_name)
    if canonical_normalized_filename in file_to_id_to_breakpoint:
        id_to_pybreakpoint = file_to_id_to_breakpoint[canonical_normalized_filename]
    else:
        id_to_pybreakpoint = file_to_id_to_breakpoint[canonical_normalized_filename] = {}
    id_to_pybreakpoint[breakpoint_id] = added_breakpoint
    py_db.consolidate_breakpoints(canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
    if py_db.plugin is not None:
        py_db.has_plugin_line_breaks = py_db.plugin.has_line_breaks()
        py_db.plugin.after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
    py_db.on_breakpoints_changed()
    return result