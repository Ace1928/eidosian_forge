import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
class ControlStatusCtx(object):
    """A context that tracks whether autograph is enabled by the user."""

    def __init__(self, status, options=None):
        self.status = status
        self.options = options

    def __enter__(self):
        _control_ctx().append(self)
        return self

    def __repr__(self):
        return '{}[status={}, options={}]'.format(self.__class__.__name__, self.status, self.options)

    def __exit__(self, unused_type, unused_value, unused_traceback):
        assert _control_ctx()[-1] is self
        _control_ctx().pop()