from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersCreateRequest(_messages.Message):
    """A CloudresourcemanagerFoldersCreateRequest object.

  Fields:
    folder: A Folder resource to be passed as the request body.
    parent: The resource name of the new Folder's parent. Must be of the form
      `folders/{folder_id}` or `organizations/{org_id}`.
  """
    folder = _messages.MessageField('Folder', 1)
    parent = _messages.StringField(2)