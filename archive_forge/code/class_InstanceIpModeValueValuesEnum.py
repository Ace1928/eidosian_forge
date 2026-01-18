from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceIpModeValueValuesEnum(_messages.Enum):
    """The IP mode for instances. Only applicable in the App Engine flexible
    environment.

    Values:
      INSTANCE_IP_MODE_UNSPECIFIED: Unspecified is treated as EXTERNAL.
      EXTERNAL: Instances are created with both internal and external IP
        addresses.
      INTERNAL: Instances are created with internal IP addresses only.
    """
    INSTANCE_IP_MODE_UNSPECIFIED = 0
    EXTERNAL = 1
    INTERNAL = 2