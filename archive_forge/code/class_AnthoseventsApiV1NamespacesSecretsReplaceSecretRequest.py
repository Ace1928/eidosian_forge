from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesSecretsReplaceSecretRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesSecretsReplaceSecretRequest object.

  Fields:
    name: Required. The name of the secret being retrieved. If needed, replace
      {namespace_id} with the project ID.
    secret: A Secret resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    secret = _messages.MessageField('Secret', 2)