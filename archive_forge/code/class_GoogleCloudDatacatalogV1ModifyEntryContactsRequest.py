from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ModifyEntryContactsRequest(_messages.Message):
    """Request message for ModifyEntryContacts.

  Fields:
    contacts: Required. The new value for the Contacts.
  """
    contacts = _messages.MessageField('GoogleCloudDatacatalogV1Contacts', 1)