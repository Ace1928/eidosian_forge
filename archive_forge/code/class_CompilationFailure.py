import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class CompilationFailure(TaskFlowException):
    """Raised when some type of compilation issue is found."""