from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsGetRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Secret to get, in the form: "proj
      ects/{projectId}/locations/{locationId}/environments/{environmentId}/use
      rWorkloadsSecrets/{userWorkloadsSecretId}"
  """
    name = _messages.StringField(1, required=True)