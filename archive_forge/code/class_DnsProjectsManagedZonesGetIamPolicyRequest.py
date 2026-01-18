from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsProjectsManagedZonesGetIamPolicyRequest(_messages.Message):
    """A DnsProjectsManagedZonesGetIamPolicyRequest object.

  Fields:
    googleIamV1GetIamPolicyRequest: A GoogleIamV1GetIamPolicyRequest resource
      to be passed as the request body.
    resource: REQUIRED: The resource for which the policy is being requested.
      See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    googleIamV1GetIamPolicyRequest = _messages.MessageField('GoogleIamV1GetIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)