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
def _convert_rules_to_exclude_filters(rules, on_error):
    exclude_filters = []
    if not isinstance(rules, list):
        on_error('Invalid "rules" (expected list of dicts). Found: %s' % (rules,))
    else:
        directory_exclude_filters = []
        module_exclude_filters = []
        glob_exclude_filters = []
        for rule in rules:
            if not isinstance(rule, dict):
                on_error('Invalid "rules" (expected list of dicts). Found: %s' % (rules,))
                continue
            include = rule.get('include')
            if include is None:
                on_error('Invalid "rule" (expected dict with "include"). Found: %s' % (rule,))
                continue
            path = rule.get('path')
            module = rule.get('module')
            if path is None and module is None:
                on_error('Invalid "rule" (expected dict with "path" or "module"). Found: %s' % (rule,))
                continue
            if path is not None:
                glob_pattern = path
                if '*' not in path and '?' not in path:
                    if os.path.isdir(glob_pattern):
                        if not glob_pattern.endswith('/') and (not glob_pattern.endswith('\\')):
                            glob_pattern += '/'
                        glob_pattern += '**'
                    directory_exclude_filters.append(ExcludeFilter(glob_pattern, not include, True))
                else:
                    glob_exclude_filters.append(ExcludeFilter(glob_pattern, not include, True))
            elif module is not None:
                module_exclude_filters.append(ExcludeFilter(module, not include, False))
            else:
                on_error('Internal error: expected path or module to be specified.')
        directory_exclude_filters = sorted(directory_exclude_filters, key=lambda exclude_filter: -len(exclude_filter.name))
        module_exclude_filters = sorted(module_exclude_filters, key=lambda exclude_filter: -len(exclude_filter.name))
        exclude_filters = directory_exclude_filters + glob_exclude_filters + module_exclude_filters
    return exclude_filters