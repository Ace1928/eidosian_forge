import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class UnclaimableJob(JobFailure):
    """Raised when a job can not be claimed."""