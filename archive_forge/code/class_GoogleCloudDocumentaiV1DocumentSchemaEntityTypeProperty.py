from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentSchemaEntityTypeProperty(_messages.Message):
    """Defines properties that can be part of the entity type.

  Enums:
    OccurrenceTypeValueValuesEnum: Occurrence type limits the number of
      instances an entity type appears in the document.

  Fields:
    displayName: User defined name for the property.
    name: The name of the property. Follows the same guidelines as the
      EntityType name.
    occurrenceType: Occurrence type limits the number of instances an entity
      type appears in the document.
    valueType: A reference to the value type of the property. This type is
      subject to the same conventions as the `Entity.base_types` field.
  """

    class OccurrenceTypeValueValuesEnum(_messages.Enum):
        """Occurrence type limits the number of instances an entity type appears
    in the document.

    Values:
      OCCURRENCE_TYPE_UNSPECIFIED: Unspecified occurrence type.
      OPTIONAL_ONCE: There will be zero or one instance of this entity type.
        The same entity instance may be mentioned multiple times.
      OPTIONAL_MULTIPLE: The entity type will appear zero or multiple times.
      REQUIRED_ONCE: The entity type will only appear exactly once. The same
        entity instance may be mentioned multiple times.
      REQUIRED_MULTIPLE: The entity type will appear once or more times.
    """
        OCCURRENCE_TYPE_UNSPECIFIED = 0
        OPTIONAL_ONCE = 1
        OPTIONAL_MULTIPLE = 2
        REQUIRED_ONCE = 3
        REQUIRED_MULTIPLE = 4
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    occurrenceType = _messages.EnumField('OccurrenceTypeValueValuesEnum', 3)
    valueType = _messages.StringField(4)