from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersGetEffectiveOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerFoldersGetEffectiveOrgPolicyRequest object.

  Fields:
    foldersId: Part of `resource`. The name of the resource to start computing
      the effective `Policy`.
    getEffectiveOrgPolicyRequest: A GetEffectiveOrgPolicyRequest resource to
      be passed as the request body.
  """
    foldersId = _messages.StringField(1, required=True)
    getEffectiveOrgPolicyRequest = _messages.MessageField('GetEffectiveOrgPolicyRequest', 2)