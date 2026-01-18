from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
def _change_variable(self, frame, name, value):
    in_vars_or_parents = False
    if 'context' in frame.f_locals:
        if name in frame.f_locals['context'].parent:
            self.back_context.parent[name] = value
            in_vars_or_parents = True
        if name in frame.f_locals['context'].vars:
            self.back_context.vars[name] = value
            in_vars_or_parents = True
    l_name = 'l_' + name
    if l_name in frame.f_locals:
        if in_vars_or_parents:
            frame.f_locals[l_name] = self.back_context.resolve(name)
        else:
            frame.f_locals[l_name] = value