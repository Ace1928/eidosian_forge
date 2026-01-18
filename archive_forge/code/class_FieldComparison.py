from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FieldComparison(_messages.Message):
    """Field that needs to be compared.

  Enums:
    ComparatorValueValuesEnum: Comparator to use for comparing the field
      value.

  Fields:
    boolValue: Boolean value
    comparator: Comparator to use for comparing the field value.
    intValue: Integer value
    key: Key of the field.
    stringValue: String value
  """

    class ComparatorValueValuesEnum(_messages.Enum):
        """Comparator to use for comparing the field value.

    Values:
      COMPARATOR_UNSPECIFIED: The default value.
      EQUALS: The field value must be equal to the specified value.
      NOT_EQUALS: The field value must not be equal to the specified value.
    """
        COMPARATOR_UNSPECIFIED = 0
        EQUALS = 1
        NOT_EQUALS = 2
    boolValue = _messages.BooleanField(1)
    comparator = _messages.EnumField('ComparatorValueValuesEnum', 2)
    intValue = _messages.IntegerField(3)
    key = _messages.StringField(4)
    stringValue = _messages.StringField(5)