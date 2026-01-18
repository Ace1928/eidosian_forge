from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposSetIamPolicyRequest(_messages.Message):
    """A SourcerepoProjectsReposSetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy is being specified.
      See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
    setIamPolicyRequest: A SetIamPolicyRequest resource to be passed as the
      request body.
  """
    resource = _messages.StringField(1, required=True)
    setIamPolicyRequest = _messages.MessageField('SetIamPolicyRequest', 2)