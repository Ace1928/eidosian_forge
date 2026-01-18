from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
class ExceptionBreakpoint(object):

    def __init__(self, qname, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries):
        exctype = get_exception_class(qname)
        self.qname = qname
        if exctype is not None:
            self.name = exctype.__name__
        else:
            self.name = None
        self.condition = condition
        self.expression = expression
        self.notify_on_unhandled_exceptions = notify_on_unhandled_exceptions
        self.notify_on_handled_exceptions = notify_on_handled_exceptions
        self.notify_on_first_raise_only = notify_on_first_raise_only
        self.notify_on_user_unhandled_exceptions = notify_on_user_unhandled_exceptions
        self.ignore_libraries = ignore_libraries
        self.type = exctype

    def __str__(self):
        return self.qname

    @property
    def has_condition(self):
        return self.condition is not None

    def handle_hit_condition(self, frame):
        return False