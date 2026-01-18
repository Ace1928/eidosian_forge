from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretEnvVar(_messages.Message):
    """Configuration for a secret environment variable. It has the information
  necessary to fetch the secret value from secret manager and expose it as an
  environment variable.

  Fields:
    key: Name of the environment variable.
    projectId: Project identifier (preferably project number but can also be
      the project ID) of the project that contains the secret. If not set, it
      is assumed that the secret is in the same project as the function.
    secret: Name of the secret in secret manager (not the full resource name).
    version: Version of the secret (version number or the string 'latest'). It
      is recommended to use a numeric version for secret environment variables
      as any updates to the secret value is not reflected until new instances
      start.
  """
    key = _messages.StringField(1)
    projectId = _messages.StringField(2)
    secret = _messages.StringField(3)
    version = _messages.StringField(4)