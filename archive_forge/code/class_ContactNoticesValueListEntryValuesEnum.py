from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContactNoticesValueListEntryValuesEnum(_messages.Enum):
    """ContactNoticesValueListEntryValuesEnum enum type.

    Values:
      CONTACT_NOTICE_UNSPECIFIED: The notice is undefined.
      PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT: Required when setting the `privacy`
        field of `ContactSettings` to `PUBLIC_CONTACT_DATA`, which exposes
        contact data publicly.
    """
    CONTACT_NOTICE_UNSPECIFIED = 0
    PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT = 1