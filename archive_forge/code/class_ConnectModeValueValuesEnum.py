from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectModeValueValuesEnum(_messages.Enum):
    """The network connect mode of the Filestore instance. If not provided,
    the connect mode defaults to DIRECT_PEERING.

    Values:
      CONNECT_MODE_UNSPECIFIED: Not set.
      DIRECT_PEERING: Connect via direct peering to the Filestore service.
      PRIVATE_SERVICE_ACCESS: Connect to your Filestore instance using Private
        Service Access. Private services access provides an IP address range
        for multiple Google Cloud services, including Filestore.
    """
    CONNECT_MODE_UNSPECIFIED = 0
    DIRECT_PEERING = 1
    PRIVATE_SERVICE_ACCESS = 2