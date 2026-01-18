from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesValidateRequest(_messages.Message):
    """A ServicenetworkingServicesValidateRequest object.

  Fields:
    parent: Required. This is in a form services/{service} where {service} is
      the name of the private access management service. For example 'service-
      peering.example.com'.
    validateConsumerConfigRequest: A ValidateConsumerConfigRequest resource to
      be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    validateConsumerConfigRequest = _messages.MessageField('ValidateConsumerConfigRequest', 2)