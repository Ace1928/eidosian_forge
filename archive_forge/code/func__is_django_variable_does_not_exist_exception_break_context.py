import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
def _is_django_variable_does_not_exist_exception_break_context(frame):
    try:
        name = frame.f_code.co_name
    except:
        name = None
    return name in ('_resolve_lookup', 'find_template')