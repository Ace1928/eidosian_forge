from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeCacheKeyset(_messages.Message):
    """Represents a collection of public keys used for validating signed
  requests.

  Messages:
    LabelsValue: Optional. A set of label tags associated with the
      EdgeCacheKeyset resource.

  Fields:
    createTime: Output only. The creation timestamp in RFC3339 text format.
    description: Optional. A human-readable description of the resource.
    labels: Optional. A set of label tags associated with the EdgeCacheKeyset
      resource.
    name: Required. The name of the resource as provided by the client when
      the resource is created. The name must be 1-64 characters long, and
      match the regular expression `[a-zA-Z]([a-zA-Z0-9_-])*` which means the
      first character must be a letter, and all following characters must be a
      dash, an underscore, a letter, or a digit.
    publicKeys: Optional. An ordered list of Ed25519 public keys to use for
      validating signed requests. Ed25519 public keys are not secret and only
      allow Google to validate that a request was signed by your corresponding
      private key. Ensure that the private key is kept secret and that only
      authorized users can add public keys to a keyset. You can rotate keys by
      appending (pushing) a new key to the list of public keys, and removing
      any superseded keys. You must specify `public_keys` or
      validation_shared_keys (or both). The keys in `public_keys` are checked
      first. You can specify at most one Google-managed public key. If you
      specify `public_keys`, you must specify at least one key and can specify
      up to three keys.
    updateTime: Output only. The update timestamp in RFC3339 text format.
    validationSharedKeys: Optional. An ordered list of shared keys to use for
      validating signed requests. Shared keys are secret. Ensure that only
      authorized users can add `validation_shared_keys` to a keyset. You can
      rotate keys by appending (pushing) a new key to the list of
      `validation_shared_keys` and removing any superseded keys. You must
      specify public_keys or `validation_shared_keys` (or both). The keys in
      `public_keys` are checked first. If you specify
      `validation_shared_keys`, you must specify at least one key and can
      specify up to three keys.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A set of label tags associated with the EdgeCacheKeyset
    resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    publicKeys = _messages.MessageField('PublicKey', 5, repeated=True)
    updateTime = _messages.StringField(6)
    validationSharedKeys = _messages.MessageField('Secret', 7, repeated=True)