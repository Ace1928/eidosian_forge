from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageConsumerPoliciesPatchRequest(_messages.Message):
    """A ServiceusageConsumerPoliciesPatchRequest object.

  Fields:
    force: This flag will skip the breaking change detections.
    googleApiServiceusageV2alphaConsumerPolicy: A
      GoogleApiServiceusageV2alphaConsumerPolicy resource to be passed as the
      request body.
    name: Output only. The resource name of the policy. Only the `default`
      policy is supported: `projects/12345/consumerPolicies/default`,
      `folders/12345/consumerPolicies/default`,
      `organizations/12345/consumerPolicies/default`.
    validateOnly: If set, validate the request and preview the result but do
      not actually commit it.
  """
    force = _messages.BooleanField(1)
    googleApiServiceusageV2alphaConsumerPolicy = _messages.MessageField('GoogleApiServiceusageV2alphaConsumerPolicy', 2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)