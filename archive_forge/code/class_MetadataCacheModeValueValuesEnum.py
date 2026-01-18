from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataCacheModeValueValuesEnum(_messages.Enum):
    """Optional. Metadata Cache Mode for the table. Set this to enable
    caching of metadata from external data source.

    Values:
      METADATA_CACHE_MODE_UNSPECIFIED: Unspecified metadata cache mode.
      AUTOMATIC: Set this mode to trigger automatic background refresh of
        metadata cache from the external source. Queries will use the latest
        available cache version within the table's maxStaleness interval.
      MANUAL: Set this mode to enable triggering manual refresh of the
        metadata cache from external source. Queries will use the latest
        manually triggered cache version within the table's maxStaleness
        interval.
    """
    METADATA_CACHE_MODE_UNSPECIFIED = 0
    AUTOMATIC = 1
    MANUAL = 2