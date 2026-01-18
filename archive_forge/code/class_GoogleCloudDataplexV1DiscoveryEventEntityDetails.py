from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DiscoveryEventEntityDetails(_messages.Message):
    """Details about the entity.

  Enums:
    TypeValueValuesEnum: The type of the entity resource.

  Fields:
    entity: The name of the entity resource. The name is the fully-qualified
      resource name.
    type: The type of the entity resource.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the entity resource.

    Values:
      ENTITY_TYPE_UNSPECIFIED: An unspecified event type.
      TABLE: Entities representing structured data.
      FILESET: Entities representing unstructured data.
    """
        ENTITY_TYPE_UNSPECIFIED = 0
        TABLE = 1
        FILESET = 2
    entity = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)