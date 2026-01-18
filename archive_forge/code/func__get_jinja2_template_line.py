from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
def _get_jinja2_template_line(frame):
    debug_info = _get_jinja2_template_debug_info(frame)
    if debug_info is None:
        return None
    lineno = frame.f_lineno
    for pair in debug_info:
        if pair[1] == lineno:
            return pair[0]
    return None