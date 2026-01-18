import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
class NullCtx(object):
    """Helper substitute for contextlib.nullcontext."""

    def __enter__(self):
        pass

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass