import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
@staticmethod
def _check_maximum(count, maximum, msg):
    """Check a count against a maximum.

        Unless maximum is -1 which indicates that there is no limit.
        """
    if maximum != -1 and count > maximum:
        raise exception.StackValidationFailed(message=msg)