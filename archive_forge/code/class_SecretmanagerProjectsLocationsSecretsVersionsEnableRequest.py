from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsVersionsEnableRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsVersionsEnableRequest object.

  Fields:
    enableSecretVersionRequest: A EnableSecretVersionRequest resource to be
      passed as the request body.
    name: Required. The resource name of the SecretVersion to enable in the
      format `projects/*/secrets/*/versions/*` or
      `projects/*/locations/*/secrets/*/versions/*`.
  """
    enableSecretVersionRequest = _messages.MessageField('EnableSecretVersionRequest', 1)
    name = _messages.StringField(2, required=True)