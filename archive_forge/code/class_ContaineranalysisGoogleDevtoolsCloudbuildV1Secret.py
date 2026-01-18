from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Secret(_messages.Message):
    """Pairs a set of secret environment variables containing encrypted values
  with the Cloud KMS key to use to decrypt the value. Note: Use `kmsKeyName`
  with `available_secrets` instead of using `kmsKeyName` with `secret`. For
  instructions see: https://cloud.google.com/cloud-build/docs/securing-
  builds/use-encrypted-credentials.

  Messages:
    SecretEnvValue: Map of environment variable name to its encrypted value.
      Secret environment variables must be unique across all of a build's
      secrets, and must be used by at least one build step. Values can be at
      most 64 KB in size. There can be at most 100 secret values across all of
      a build's secrets.

  Fields:
    kmsKeyName: Cloud KMS key name to use to decrypt these envs.
    secretEnv: Map of environment variable name to its encrypted value. Secret
      environment variables must be unique across all of a build's secrets,
      and must be used by at least one build step. Values can be at most 64 KB
      in size. There can be at most 100 secret values across all of a build's
      secrets.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SecretEnvValue(_messages.Message):
        """Map of environment variable name to its encrypted value. Secret
    environment variables must be unique across all of a build's secrets, and
    must be used by at least one build step. Values can be at most 64 KB in
    size. There can be at most 100 secret values across all of a build's
    secrets.

    Messages:
      AdditionalProperty: An additional property for a SecretEnvValue object.

    Fields:
      additionalProperties: Additional properties of type SecretEnvValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SecretEnvValue object.

      Fields:
        key: Name of the additional property.
        value: A byte attribute.
      """
            key = _messages.StringField(1)
            value = _messages.BytesField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    kmsKeyName = _messages.StringField(1)
    secretEnv = _messages.MessageField('SecretEnvValue', 2)