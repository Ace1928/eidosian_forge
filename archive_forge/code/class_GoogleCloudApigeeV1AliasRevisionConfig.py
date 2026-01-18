from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AliasRevisionConfig(_messages.Message):
    """A GoogleCloudApigeeV1AliasRevisionConfig object.

  Enums:
    TypeValueValuesEnum:

  Fields:
    location: Location of the alias file. For example, a Google Cloud Storage
      URI.
    name: Name of the alias revision included in the keystore in the following
      format: `organizations/{org}/environments/{env}/keystores/{keystore}/ali
      ases/{alias}/revisions/{rev}`
    type: A TypeValueValuesEnum attribute.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """TypeValueValuesEnum enum type.

    Values:
      ALIAS_TYPE_UNSPECIFIED: Alias type is not specified.
      CERT: Certificate.
      KEY_CERT: Key/certificate pair.
    """
        ALIAS_TYPE_UNSPECIFIED = 0
        CERT = 1
        KEY_CERT = 2
    location = _messages.StringField(1)
    name = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)