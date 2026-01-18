from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1GovernanceEventEntity(_messages.Message):
    """Information about Entity resource that the log event is associated with.

  Enums:
    EntityTypeValueValuesEnum: Type of entity.

  Fields:
    entity: The Entity resource the log event is associated with. Format: proj
      ects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{zon
      e_id}/entities/{entity_id}
    entityType: Type of entity.
  """

    class EntityTypeValueValuesEnum(_messages.Enum):
        """Type of entity.

    Values:
      ENTITY_TYPE_UNSPECIFIED: An unspecified Entity type.
      TABLE: Table entity type.
      FILESET: Fileset entity type.
    """
        ENTITY_TYPE_UNSPECIFIED = 0
        TABLE = 1
        FILESET = 2
    entity = _messages.StringField(1)
    entityType = _messages.EnumField('EntityTypeValueValuesEnum', 2)