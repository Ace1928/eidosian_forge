from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageManagedFoldersSetIamPolicyRequest(_messages.Message):
    """A StorageManagedFoldersSetIamPolicyRequest object.

  Fields:
    bucket: Name of the bucket containing the managed folder.
    managedFolder: The managed folder name/path.
    policy: A Policy resource to be passed as the request body.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    managedFolder = _messages.StringField(2, required=True)
    policy = _messages.MessageField('Policy', 3)
    userProject = _messages.StringField(4)