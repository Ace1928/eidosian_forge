from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprConstant(_messages.Message):
    """Represents a primitive literal. Named 'Constant' here for backwards
  compatibility. This is similar as the primitives supported in the well-known
  type `google.protobuf.Value`, but richer so it can represent CEL's full
  range of primitives. Lists and structs are not included as constants as
  these aggregate types may contain Expr elements which require evaluation and
  are thus not constant. Examples of constants include: `"hello"`, `b'bytes'`,
  `1u`, `4.2`, `-2`, `true`, `null`.

  Enums:
    NullValueValueValuesEnum: null value.

  Fields:
    boolValue: boolean value.
    bytesValue: bytes value.
    doubleValue: double value.
    durationValue: protobuf.Duration value. Deprecated: duration is no longer
      considered a builtin cel type.
    int64Value: int64 value.
    nullValue: null value.
    stringValue: string value.
    timestampValue: protobuf.Timestamp value. Deprecated: timestamp is no
      longer considered a builtin cel type.
    uint64Value: uint64 value.
  """

    class NullValueValueValuesEnum(_messages.Enum):
        """null value.

    Values:
      NULL_VALUE: Null value.
    """
        NULL_VALUE = 0
    boolValue = _messages.BooleanField(1)
    bytesValue = _messages.BytesField(2)
    doubleValue = _messages.FloatField(3)
    durationValue = _messages.StringField(4)
    int64Value = _messages.IntegerField(5)
    nullValue = _messages.EnumField('NullValueValueValuesEnum', 6)
    stringValue = _messages.StringField(7)
    timestampValue = _messages.StringField(8)
    uint64Value = _messages.IntegerField(9, variant=_messages.Variant.UINT64)