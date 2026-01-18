from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageViewValueValuesEnum(_messages.Enum):
    """Specifies the parts of the Message resource to include in the export.
    If not specified, FULL is used.

    Values:
      MESSAGE_VIEW_UNSPECIFIED: Not specified, equivalent to FULL for
        getMessage, equivalent to BASIC for listMessages.
      RAW_ONLY: Server responses include all the message fields except
        parsed_data and schematized_data fields.
      PARSED_ONLY: Server responses include all the message fields except data
        and schematized_data fields.
      FULL: Server responses include all the message fields.
      SCHEMATIZED_ONLY: Server responses include all the message fields except
        data and parsed_data fields.
      BASIC: Server responses include only the name field.
    """
    MESSAGE_VIEW_UNSPECIFIED = 0
    RAW_ONLY = 1
    PARSED_ONLY = 2
    FULL = 3
    SCHEMATIZED_ONLY = 4
    BASIC = 5