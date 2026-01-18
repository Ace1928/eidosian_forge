from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesGetIamPolicyRequest(_messages.Message):
    """A ServicemanagementServicesGetIamPolicyRequest object.

  Fields:
    getIamPolicyRequest: A GetIamPolicyRequest resource to be passed as the
      request body.
    servicesId: Part of `resource`. REQUIRED: The resource for which the
      policy is being requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    getIamPolicyRequest = _messages.MessageField('GetIamPolicyRequest', 1)
    servicesId = _messages.StringField(2, required=True)