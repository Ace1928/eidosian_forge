from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log
class LineBreakpointWithLazyValidation(LineBreakpoint):

    def __init__(self, *args, **kwargs):
        LineBreakpoint.__init__(self, *args, **kwargs)
        self.add_breakpoint_result = None
        self.on_changed_breakpoint_state = None
        self.verified_cache_key = None