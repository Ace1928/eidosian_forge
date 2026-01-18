import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
def _is_django_render_call(frame, debug=False):
    try:
        name = frame.f_code.co_name
        if name != 'render':
            return False
        if 'self' not in frame.f_locals:
            return False
        cls = frame.f_locals['self'].__class__
        inherits_node = _inherits(cls, 'Node')
        if not inherits_node:
            return False
        clsname = cls.__name__
        if IS_DJANGO19:
            if clsname == 'IncludeNode':
                if 'context' in frame.f_locals:
                    context = frame.f_locals['context']
                    context._has_included_template = True
        return clsname not in _IGNORE_RENDER_OF_CLASSES
    except:
        pydev_log.exception()
        return False