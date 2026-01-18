from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MethodPolicy(_messages.Message):
    """Defines policies applying to an RPC method.

  Fields:
    requestPolicies: Policies that are applicable to the request message.
    selector: Selects a method to which these policies should be enforced, for
      example, "google.pubsub.v1.Subscriber.CreateSubscription". Refer to
      selector for syntax details. NOTE: This field must not be set in the
      proto annotation. It will be automatically filled by the service config
      compiler .
  """
    requestPolicies = _messages.MessageField('FieldPolicy', 1, repeated=True)
    selector = _messages.StringField(2)