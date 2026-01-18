import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
class DjangoLineBreakpoint(LineBreakpointWithLazyValidation):

    def __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=None, is_logpoint=False):
        self.canonical_normalized_filename = canonical_normalized_filename
        LineBreakpointWithLazyValidation.__init__(self, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)

    def __str__(self):
        return 'DjangoLineBreakpoint: %s-%d' % (self.canonical_normalized_filename, self.line)