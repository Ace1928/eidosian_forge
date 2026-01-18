import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class ConductorFailure(TaskFlowException):
    """Errors related to conducting activities."""