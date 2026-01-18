from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsGetEffectiveOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerProjectsGetEffectiveOrgPolicyRequest object.

  Fields:
    getEffectiveOrgPolicyRequest: A GetEffectiveOrgPolicyRequest resource to
      be passed as the request body.
    projectsId: Part of `resource`. The name of the resource to start
      computing the effective `Policy`.
  """
    getEffectiveOrgPolicyRequest = _messages.MessageField('GetEffectiveOrgPolicyRequest', 1)
    projectsId = _messages.StringField(2, required=True)