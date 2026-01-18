from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsVersionsDisableRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsVersionsDisableRequest object.

  Fields:
    disableSecretVersionRequest: A DisableSecretVersionRequest resource to be
      passed as the request body.
    name: Required. The resource name of the SecretVersion to disable in the
      format `projects/*/secrets/*/versions/*` or
      `projects/*/locations/*/secrets/*/versions/*`.
  """
    disableSecretVersionRequest = _messages.MessageField('DisableSecretVersionRequest', 1)
    name = _messages.StringField(2, required=True)