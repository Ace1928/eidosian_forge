from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotTypeValueValuesEnum(_messages.Enum):
    """Indicates the type of the snapshot.

    Values:
      ARCHIVE: <no description>
      STANDARD: <no description>
    """
    ARCHIVE = 0
    STANDARD = 1