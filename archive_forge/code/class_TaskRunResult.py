from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskRunResult(_messages.Message):
    """TaskRunResult used to describe the results of a task

  Enums:
    TypeValueValuesEnum: The type of data that the result holds.

  Fields:
    name: Name of the TaskRun
    resultValue: Value of the result.
    type: The type of data that the result holds.
    value: Value of the result. Deprecated; please use result_value instead.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of data that the result holds.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
      ARRAY: Array type
      OBJECT: Object type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        ARRAY = 2
        OBJECT = 3
    name = _messages.StringField(1)
    resultValue = _messages.MessageField('ResultValue', 2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)
    value = _messages.StringField(4)