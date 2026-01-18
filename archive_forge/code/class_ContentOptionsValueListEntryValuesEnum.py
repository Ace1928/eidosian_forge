from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContentOptionsValueListEntryValuesEnum(_messages.Enum):
    """ContentOptionsValueListEntryValuesEnum enum type.

    Values:
      CONTENT_UNSPECIFIED: Includes entire content of a file or a data stream.
      CONTENT_TEXT: Text content within the data, excluding any metadata.
      CONTENT_IMAGE: Images found in the data.
    """
    CONTENT_UNSPECIFIED = 0
    CONTENT_TEXT = 1
    CONTENT_IMAGE = 2