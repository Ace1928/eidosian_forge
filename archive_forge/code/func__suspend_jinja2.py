from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
def _suspend_jinja2(pydb, thread, frame, cmd=CMD_SET_BREAK, message=None):
    frame = Jinja2TemplateFrame(frame)
    if frame.f_lineno is None:
        return None
    pydb.set_suspend(thread, cmd)
    thread.additional_info.suspend_type = JINJA2_SUSPEND
    if cmd == CMD_ADD_EXCEPTION_BREAK:
        if message:
            message = str(message)
        thread.additional_info.pydev_message = message
    return frame