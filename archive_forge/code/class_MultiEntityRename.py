from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiEntityRename(_messages.Message):
    """Options to configure rule type MultiEntityRename. The rule is used to
  rename multiple entities. The rule filter field can refer to one or more
  entities. The rule scope can be one of: Database, Schema, Table, Column,
  Constraint, Index, View, Function, Stored Procedure, Materialized View,
  Sequence, UDT

  Enums:
    SourceNameTransformationValueValuesEnum: Optional. Additional
      transformation that can be done on the source entity name before it is
      being used by the new_name_pattern, for example lower case. If no
      transformation is desired, use NO_TRANSFORMATION

  Fields:
    newNamePattern: Optional. The pattern used to generate the new entity's
      name. This pattern must include the characters '{name}', which will be
      replaced with the name of the original entity. For example, the pattern
      't_{name}' for an entity name jobs would be converted to 't_jobs'. If
      unspecified, the default value for this field is '{name}'
    sourceNameTransformation: Optional. Additional transformation that can be
      done on the source entity name before it is being used by the
      new_name_pattern, for example lower case. If no transformation is
      desired, use NO_TRANSFORMATION
  """

    class SourceNameTransformationValueValuesEnum(_messages.Enum):
        """Optional. Additional transformation that can be done on the source
    entity name before it is being used by the new_name_pattern, for example
    lower case. If no transformation is desired, use NO_TRANSFORMATION

    Values:
      ENTITY_NAME_TRANSFORMATION_UNSPECIFIED: Entity name transformation
        unspecified.
      ENTITY_NAME_TRANSFORMATION_NO_TRANSFORMATION: No transformation.
      ENTITY_NAME_TRANSFORMATION_LOWER_CASE: Transform to lower case.
      ENTITY_NAME_TRANSFORMATION_UPPER_CASE: Transform to upper case.
      ENTITY_NAME_TRANSFORMATION_CAPITALIZED_CASE: Transform to capitalized
        case.
    """
        ENTITY_NAME_TRANSFORMATION_UNSPECIFIED = 0
        ENTITY_NAME_TRANSFORMATION_NO_TRANSFORMATION = 1
        ENTITY_NAME_TRANSFORMATION_LOWER_CASE = 2
        ENTITY_NAME_TRANSFORMATION_UPPER_CASE = 3
        ENTITY_NAME_TRANSFORMATION_CAPITALIZED_CASE = 4
    newNamePattern = _messages.StringField(1)
    sourceNameTransformation = _messages.EnumField('SourceNameTransformationValueValuesEnum', 2)