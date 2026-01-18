import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
def _control_ctx():
    if not hasattr(stacks, 'control_status'):
        stacks.control_status = [_default_control_status_ctx()]
    return stacks.control_status