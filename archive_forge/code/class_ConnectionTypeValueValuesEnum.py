from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionTypeValueValuesEnum(_messages.Enum):
    """Required. The other-cloud connection type.

    Values:
      CONNECTION_TYPE_UNSPECIFIED: Connection type unspecified.
      COLLECT_AWS_ASSET: Collects asset config data from AWS.
    """
    CONNECTION_TYPE_UNSPECIFIED = 0
    COLLECT_AWS_ASSET = 1