from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsFoldersContactsDeleteRequest(_messages.Message):
    """A EssentialcontactsFoldersContactsDeleteRequest object.

  Fields:
    name: Required. The name of the contact to delete. Format:
      organizations/{organization_id}/contacts/{contact_id},
      folders/{folder_id}/contacts/{contact_id} or
      projects/{project_id}/contacts/{contact_id}
  """
    name = _messages.StringField(1, required=True)