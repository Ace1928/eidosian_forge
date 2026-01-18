import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
def _get_original_filename_from_origin_in_parent_frame_locals(frame, parent_frame_name):
    filename = None
    parent_frame = frame
    while parent_frame.f_code.co_name != parent_frame_name:
        parent_frame = parent_frame.f_back
    origin = None
    if parent_frame is not None:
        origin = parent_frame.f_locals.get('origin')
    if hasattr(origin, 'name') and origin.name is not None:
        filename = _convert_to_str(origin.name)
    return filename