from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HostTypeValueValuesEnum(_messages.Enum):
    """Required. The type of hosting used by the AppGateway.

    Values:
      HOST_TYPE_UNSPECIFIED: Default value. This value is unused.
      GCP_REGIONAL_MIG: AppGateway hosted in a GCP regional managed instance
        group.
    """
    HOST_TYPE_UNSPECIFIED = 0
    GCP_REGIONAL_MIG = 1