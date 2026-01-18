from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsSetOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerProjectsSetOrgPolicyRequest object.

  Fields:
    projectsId: Part of `resource`. Resource name of the resource to attach
      the `Policy`.
    setOrgPolicyRequest: A SetOrgPolicyRequest resource to be passed as the
      request body.
  """
    projectsId = _messages.StringField(1, required=True)
    setOrgPolicyRequest = _messages.MessageField('SetOrgPolicyRequest', 2)