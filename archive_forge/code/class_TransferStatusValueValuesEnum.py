from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferStatusValueValuesEnum(_messages.Enum):
    """Output only. Denotes the transfer status of the Entry Group. It is
    unspecified for Entry Group created from Dataplex API.

    Values:
      TRANSFER_STATUS_UNSPECIFIED: The default value. It is set for resources
        that were not subject for migration from Data Catalog service.
      TRANSFER_STATUS_MIGRATED: Indicates that a resource was migrated from
        Data Catalog service but it hasn't been transferred yet. In particular
        the resource cannot be updated from Dataplex API.
      TRANSFER_STATUS_TRANSFERRED: Indicates that a resource was transferred
        from Data Catalog service. The resource can only be updated from
        Dataplex API.
    """
    TRANSFER_STATUS_UNSPECIFIED = 0
    TRANSFER_STATUS_MIGRATED = 1
    TRANSFER_STATUS_TRANSFERRED = 2