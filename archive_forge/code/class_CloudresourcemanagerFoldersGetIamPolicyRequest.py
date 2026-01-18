from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersGetIamPolicyRequest(_messages.Message):
    """A CloudresourcemanagerFoldersGetIamPolicyRequest object.

  Fields:
    foldersId: Part of `resource`. REQUIRED: The resource for which the policy
      is being requested. See the operation documentation for the appropriate
      value for this field.
    getIamPolicyRequest: A GetIamPolicyRequest resource to be passed as the
      request body.
  """
    foldersId = _messages.StringField(1, required=True)
    getIamPolicyRequest = _messages.MessageField('GetIamPolicyRequest', 2)