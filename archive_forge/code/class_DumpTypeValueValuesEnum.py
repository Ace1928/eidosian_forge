from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DumpTypeValueValuesEnum(_messages.Enum):
    """Optional. The type of the data dump. Supported for MySQL to CloudSQL
    for MySQL migrations only.

    Values:
      DUMP_TYPE_UNSPECIFIED: If not specified, defaults to LOGICAL
      LOGICAL: Logical dump.
      PHYSICAL: Physical file-based dump. Supported for MySQL to CloudSQL for
        MySQL migrations only.
    """
    DUMP_TYPE_UNSPECIFIED = 0
    LOGICAL = 1
    PHYSICAL = 2