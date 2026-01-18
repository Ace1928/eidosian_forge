import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
class Status(enum.Enum):
    UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2