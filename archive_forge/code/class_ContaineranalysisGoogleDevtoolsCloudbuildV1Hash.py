from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Hash(_messages.Message):
    """Container message for hash values.

  Enums:
    TypeValueValuesEnum: The type of hash that was performed.

  Fields:
    type: The type of hash that was performed.
    value: The hash value.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of hash that was performed.

    Values:
      NONE: No hash requested.
      SHA256: Use a sha256 hash.
      MD5: Use a md5 hash.
      SHA512: Use a sha512 hash.
    """
        NONE = 0
        SHA256 = 1
        MD5 = 2
        SHA512 = 3
    type = _messages.EnumField('TypeValueValuesEnum', 1)
    value = _messages.BytesField(2)