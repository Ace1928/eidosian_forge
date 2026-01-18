from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ConfigMap(_messages.Message):
    """Cloud Run fully managed: not supported Cloud Run for Anthos: supported
  ConfigMap holds configuration data for pods to consume.

  Messages:
    BinaryDataValue: BinaryData contains the binary data. Each key must
      consist of alphanumeric characters, '-', '_' or '.'. BinaryData can
      contain byte sequences that are not in the UTF-8 range. The keys stored
      in BinaryData must not overlap with the ones in the Data field, this is
      enforced during validation process. Using this field will require 1.10+
      apiserver and kubelet.
    DataValue: Data contains the configuration data. Each key must consist of
      alphanumeric characters, '-', '_' or '.'. Values with non-UTF-8 byte
      sequences must use the BinaryData field. The keys stored in Data must
      not overlap with the keys in the BinaryData field, this is enforced
      during validation process.

  Fields:
    binaryData: BinaryData contains the binary data. Each key must consist of
      alphanumeric characters, '-', '_' or '.'. BinaryData can contain byte
      sequences that are not in the UTF-8 range. The keys stored in BinaryData
      must not overlap with the ones in the Data field, this is enforced
      during validation process. Using this field will require 1.10+ apiserver
      and kubelet.
    data: Data contains the configuration data. Each key must consist of
      alphanumeric characters, '-', '_' or '.'. Values with non-UTF-8 byte
      sequences must use the BinaryData field. The keys stored in Data must
      not overlap with the keys in the BinaryData field, this is enforced
      during validation process.
    immutable: Immutable, if set to true, ensures that data stored in the
      ConfigMap cannot be updated (only object metadata can be modified). If
      not set to true, the field can be modified at any time. Defaulted to
      nil. This is a beta field enabled by ImmutableEphemeralVolumes feature
      gate.
    metadata: Standard object's metadata. More info:
      https://git.k8s.io/community/contributors/devel/sig-architecture/api-
      conventions.md#metadata
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BinaryDataValue(_messages.Message):
        """BinaryData contains the binary data. Each key must consist of
    alphanumeric characters, '-', '_' or '.'. BinaryData can contain byte
    sequences that are not in the UTF-8 range. The keys stored in BinaryData
    must not overlap with the ones in the Data field, this is enforced during
    validation process. Using this field will require 1.10+ apiserver and
    kubelet.

    Messages:
      AdditionalProperty: An additional property for a BinaryDataValue object.

    Fields:
      additionalProperties: Additional properties of type BinaryDataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BinaryDataValue object.

      Fields:
        key: Name of the additional property.
        value: A byte attribute.
      """
            key = _messages.StringField(1)
            value = _messages.BytesField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """Data contains the configuration data. Each key must consist of
    alphanumeric characters, '-', '_' or '.'. Values with non-UTF-8 byte
    sequences must use the BinaryData field. The keys stored in Data must not
    overlap with the keys in the BinaryData field, this is enforced during
    validation process.

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Additional properties of type DataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    binaryData = _messages.MessageField('BinaryDataValue', 1)
    data = _messages.MessageField('DataValue', 2)
    immutable = _messages.BooleanField(3)
    metadata = _messages.MessageField('ObjectMeta', 4)