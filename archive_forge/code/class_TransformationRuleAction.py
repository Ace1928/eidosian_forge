from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformationRuleAction(_messages.Message):
    """TransformationRuleAction defines a TransformationRule action based on
  the JSON Patch RFC (https://www.rfc-editor.org/rfc/rfc6902)

  Enums:
    OpValueValuesEnum: Required. op specifies the operation to perform.

  Fields:
    fromPath: Optional. A string containing a JSON Pointer value that
      references the location in the target document to move the value from.
    op: Required. op specifies the operation to perform.
    path: Optional. A string containing a JSON-Pointer value that references a
      location within the target document where the operation is performed.
    value: Optional. A string that specifies the desired value in string
      format to use for transformation.
  """

    class OpValueValuesEnum(_messages.Enum):
        """Required. op specifies the operation to perform.

    Values:
      OP_UNSPECIFIED: Unspecified operation
      REMOVE: The "remove" operation removes the value at the target location.
      MOVE: The "move" operation removes the value at a specified location and
        adds it to the target location.
      COPY: The "copy" operation copies the value at a specified location to
        the target location.
      ADD: The "add" operation performs one of the following functions,
        depending upon what the target location references: 1. If the target
        location specifies an array index, a new value is inserted into the
        array at the specified index. 2. If the target location specifies an
        object member that does not already exist, a new member is added to
        the object. 3. If the target location specifies an object member that
        does exist, that member's value is replaced.
      TEST: The "test" operation tests that a value at the target location is
        equal to a specified value.
      REPLACE: The "replace" operation replaces the value at the target
        location with a new value. The operation object MUST contain a "value"
        member whose content specifies the replacement value.
    """
        OP_UNSPECIFIED = 0
        REMOVE = 1
        MOVE = 2
        COPY = 3
        ADD = 4
        TEST = 5
        REPLACE = 6
    fromPath = _messages.StringField(1)
    op = _messages.EnumField('OpValueValuesEnum', 2)
    path = _messages.StringField(3)
    value = _messages.StringField(4)