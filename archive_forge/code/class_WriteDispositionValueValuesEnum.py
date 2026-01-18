from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteDispositionValueValuesEnum(_messages.Enum):
    """Determines if existing data in the destination dataset is overwritten,
    appended to, or not written if the tables contain data. If a
    write_disposition is specified, the `force` parameter is ignored.

    Values:
      WRITE_DISPOSITION_UNSPECIFIED: Default behavior is the same as
        WRITE_EMPTY.
      WRITE_EMPTY: Only export data if the destination tables are empty.
      WRITE_TRUNCATE: Erase all existing data in the destination tables before
        writing the FHIR resources.
      WRITE_APPEND: Append data to the destination tables.
    """
    WRITE_DISPOSITION_UNSPECIFIED = 0
    WRITE_EMPTY = 1
    WRITE_TRUNCATE = 2
    WRITE_APPEND = 3