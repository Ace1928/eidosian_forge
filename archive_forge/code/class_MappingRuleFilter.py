from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MappingRuleFilter(_messages.Message):
    """A filter defining the entities that a mapping rule should be applied to.
  When more than one field is specified, the rule is applied only to entities
  which match all the fields.

  Fields:
    entities: Optional. The rule should be applied to specific entities
      defined by their fully qualified names.
    entityNameContains: Optional. The rule should be applied to entities whose
      non-qualified name contains the given string.
    entityNamePrefix: Optional. The rule should be applied to entities whose
      non-qualified name starts with the given prefix.
    entityNameSuffix: Optional. The rule should be applied to entities whose
      non-qualified name ends with the given suffix.
    parentEntity: Optional. The rule should be applied to entities whose
      parent entity (fully qualified name) matches the given value. For
      example, if the rule applies to a table entity, the expected value
      should be a schema (schema). If the rule applies to a column or index
      entity, the expected value can be either a schema (schema) or a table
      (schema.table)
  """
    entities = _messages.StringField(1, repeated=True)
    entityNameContains = _messages.StringField(2)
    entityNamePrefix = _messages.StringField(3)
    entityNameSuffix = _messages.StringField(4)
    parentEntity = _messages.StringField(5)