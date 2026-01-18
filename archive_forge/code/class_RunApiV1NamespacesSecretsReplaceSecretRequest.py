from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunApiV1NamespacesSecretsReplaceSecretRequest(_messages.Message):
    """A RunApiV1NamespacesSecretsReplaceSecretRequest object.

  Fields:
    name: Required. The name of the secret being retrieved. If needed, replace
      {namespace} with the project ID or number. It takes the form
      namespaces/{namespace}. For example: namespaces/PROJECT_ID
    secret: A Secret resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    secret = _messages.MessageField('Secret', 2)