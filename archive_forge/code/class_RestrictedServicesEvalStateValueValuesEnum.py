from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestrictedServicesEvalStateValueValuesEnum(_messages.Enum):
    """Eval state of the restricted services

    Values:
      RESTRICTED_SERVICES_EVAL_STATE_UNSPECIFIED: Not used
      IS_RESTRICTED: The request service is restricted
      IS_NOT_RESTRICTED: The request service is not restricted
    """
    RESTRICTED_SERVICES_EVAL_STATE_UNSPECIFIED = 0
    IS_RESTRICTED = 1
    IS_NOT_RESTRICTED = 2