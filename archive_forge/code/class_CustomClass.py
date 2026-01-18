from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomClass(_messages.Message):
    """CustomClass for biasing in speech recognition. Used to define a set of
  words or phrases that represents a common concept or theme likely to appear
  in your audio, for example a list of passenger ship names.

  Enums:
    StateValueValuesEnum: Output only. The CustomClass lifecycle state.

  Messages:
    AnnotationsValue: Optional. Allows users to store small amounts of
      arbitrary data. Both the key and the value must be 63 characters or less
      each. At most 100 annotations.

  Fields:
    annotations: Optional. Allows users to store small amounts of arbitrary
      data. Both the key and the value must be 63 characters or less each. At
      most 100 annotations.
    createTime: Output only. Creation time.
    deleteTime: Output only. The time at which this resource was requested for
      deletion.
    displayName: Optional. User-settable, human-readable name for the
      CustomClass. Must be 63 characters or less.
    etag: Output only. This checksum is computed by the server based on the
      value of other fields. This may be sent on update, undelete, and delete
      requests to ensure the client has an up-to-date value before proceeding.
    expireTime: Output only. The time at which this resource will be purged.
    items: A collection of class items.
    kmsKeyName: Output only. The [KMS key
      name](https://cloud.google.com/kms/docs/resource-hierarchy#keys) with
      which the CustomClass is encrypted. The expected format is `projects/{pr
      oject}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}`
      .
    kmsKeyVersionName: Output only. The [KMS key version
      name](https://cloud.google.com/kms/docs/resource-hierarchy#key_versions)
      with which the CustomClass is encrypted. The expected format is `project
      s/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_
      key}/cryptoKeyVersions/{crypto_key_version}`.
    name: Output only. Identifier. The resource name of the CustomClass.
      Format:
      `projects/{project}/locations/{location}/customClasses/{custom_class}`.
    reconciling: Output only. Whether or not this CustomClass is in the
      process of being updated.
    state: Output only. The CustomClass lifecycle state.
    uid: Output only. System-assigned unique identifier for the CustomClass.
    updateTime: Output only. The most recent time this resource was modified.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The CustomClass lifecycle state.

    Values:
      STATE_UNSPECIFIED: Unspecified state. This is only used/useful for
        distinguishing unset values.
      ACTIVE: The normal and active state.
      DELETED: This CustomClass has been deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Allows users to store small amounts of arbitrary data. Both
    the key and the value must be 63 characters or less each. At most 100
    annotations.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    items = _messages.MessageField('ClassItem', 7, repeated=True)
    kmsKeyName = _messages.StringField(8)
    kmsKeyVersionName = _messages.StringField(9)
    name = _messages.StringField(10)
    reconciling = _messages.BooleanField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    uid = _messages.StringField(13)
    updateTime = _messages.StringField(14)