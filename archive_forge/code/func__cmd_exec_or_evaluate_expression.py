import json
import os
import sys
import traceback
from _pydev_bundle import pydev_log
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydevd_bundle import pydevd_traceproperty, pydevd_dont_trace, pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import get_exception_class
from _pydevd_bundle.pydevd_comm import (
from _pydevd_bundle.pydevd_constants import NEXT_VALUE_SEPARATOR, IS_WINDOWS, NULL
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXEC_EXPRESSION, CMD_AUTHENTICATE
from _pydevd_bundle.pydevd_api import PyDevdAPI
from io import StringIO
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
import pydevd_file_utils
def _cmd_exec_or_evaluate_expression(self, py_db, cmd_id, seq, text):
    attr_to_set_result = ''
    try:
        thread_id, frame_id, scope, expression, trim, attr_to_set_result = text.split('\t', 5)
    except ValueError:
        thread_id, frame_id, scope, expression, trim = text.split('\t', 4)
    is_exec = cmd_id == CMD_EXEC_EXPRESSION
    trim_if_too_big = int(trim) == 1
    self.api.request_exec_or_evaluate(py_db, seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result)