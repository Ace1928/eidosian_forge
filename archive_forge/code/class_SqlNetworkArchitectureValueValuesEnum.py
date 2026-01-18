from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlNetworkArchitectureValueValuesEnum(_messages.Enum):
    """The SQL network architecture for the instance.

    Values:
      SQL_NETWORK_ARCHITECTURE_UNSPECIFIED: <no description>
      NEW_NETWORK_ARCHITECTURE: The instance uses the new network
        architecture.
      OLD_NETWORK_ARCHITECTURE: The instance uses the old network
        architecture.
    """
    SQL_NETWORK_ARCHITECTURE_UNSPECIFIED = 0
    NEW_NETWORK_ARCHITECTURE = 1
    OLD_NETWORK_ARCHITECTURE = 2