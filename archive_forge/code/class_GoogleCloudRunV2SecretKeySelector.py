from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2SecretKeySelector(_messages.Message):
    """SecretEnvVarSource represents a source for the value of an EnvVar.

  Fields:
    secret: Required. The name of the secret in Cloud Secret Manager. Format:
      {secret_name} if the secret is in the same project.
      projects/{project}/secrets/{secret_name} if the secret is in a different
      project.
    version: The Cloud Secret Manager secret version. Can be 'latest' for the
      latest version, an integer for a specific version, or a version alias.
  """
    secret = _messages.StringField(1)
    version = _messages.StringField(2)