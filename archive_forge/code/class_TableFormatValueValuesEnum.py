from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableFormatValueValuesEnum(_messages.Enum):
    """Required. The table format the metadata only snapshots are stored in.

    Values:
      TABLE_FORMAT_UNSPECIFIED: Default Value.
      ICEBERG: Apache Iceberg format.
    """
    TABLE_FORMAT_UNSPECIFIED = 0
    ICEBERG = 1