from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendTypeValueValuesEnum(_messages.Enum):
    """Type of load balancer's backend configuration.

    Values:
      BACKEND_TYPE_UNSPECIFIED: Type is unspecified.
      BACKEND_SERVICE: Backend Service as the load balancer's backend.
      TARGET_POOL: Target Pool as the load balancer's backend.
      TARGET_INSTANCE: Target Instance as the load balancer's backend.
    """
    BACKEND_TYPE_UNSPECIFIED = 0
    BACKEND_SERVICE = 1
    TARGET_POOL = 2
    TARGET_INSTANCE = 3