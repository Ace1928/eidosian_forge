from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsAddVersionRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsAddVersionRequest object.

  Fields:
    addSecretVersionRequest: A AddSecretVersionRequest resource to be passed
      as the request body.
    parent: Required. The resource name of the Secret to associate with the
      SecretVersion in the format `projects/*/secrets/*` or
      `projects/*/locations/*/secrets/*`.
  """
    addSecretVersionRequest = _messages.MessageField('AddSecretVersionRequest', 1)
    parent = _messages.StringField(2, required=True)