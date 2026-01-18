from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsKeyDigest(_messages.Message):
    """A DnsKeyDigest object.

  Enums:
    TypeValueValuesEnum: Specifies the algorithm used to calculate this
      digest.

  Fields:
    digest: The base-16 encoded bytes of this digest. Suitable for use in a DS
      resource record.
    type: Specifies the algorithm used to calculate this digest.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Specifies the algorithm used to calculate this digest.

    Values:
      sha1: <no description>
      sha256: <no description>
      sha384: <no description>
    """
        sha1 = 0
        sha256 = 1
        sha384 = 2
    digest = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)