from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpConfigurationValueValuesEnum(_messages.Enum):
    """Configuration for VM IPs.

    Values:
      WORKER_IP_UNSPECIFIED: The configuration is unknown, or unspecified.
      WORKER_IP_PUBLIC: Workers should have public IP addresses.
      WORKER_IP_PRIVATE: Workers should have private IP addresses.
    """
    WORKER_IP_UNSPECIFIED = 0
    WORKER_IP_PUBLIC = 1
    WORKER_IP_PRIVATE = 2