import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class StorageFailure(TaskFlowException):
    """Raised when storage backends can not be read/saved/deleted."""