from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConstraintEntity(_messages.Message):
    """Constraint is not used as an independent entity, it is retrieved as part
  of another entity such as Table or View.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    customFeatures: Custom engine specific features.
    name: The name of the table constraint.
    referenceColumns: Reference columns which may be associated with the
      constraint. For example, if the constraint is a FOREIGN_KEY, this
      represents the list of full names of referenced columns by the foreign
      key.
    referenceTable: Reference table which may be associated with the
      constraint. For example, if the constraint is a FOREIGN_KEY, this
      represents the list of full name of the referenced table by the foreign
      key.
    tableColumns: Table columns used as part of the Constraint, for example
      primary key constraint should list the columns which constitutes the
      key.
    tableName: Table which is associated with the constraint. In case the
      constraint is defined on a table, this field is left empty as this
      information is stored in parent_name. However, if constraint is defined
      on a view, this field stores the table name on which the view is
      defined.
    type: Type of constraint, for example unique, primary key, foreign key
      (currently only primary key is supported).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 1)
    name = _messages.StringField(2)
    referenceColumns = _messages.StringField(3, repeated=True)
    referenceTable = _messages.StringField(4)
    tableColumns = _messages.StringField(5, repeated=True)
    tableName = _messages.StringField(6)
    type = _messages.StringField(7)