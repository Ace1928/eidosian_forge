from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesGetRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. Used to select the subset of entity
      information to return. Defaults to BASIC.

  Fields:
    name: Required. The resource name of the entity: projects/{project_number}
      /locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/entities/{entit
      y_id}.
    view: Optional. Used to select the subset of entity information to return.
      Defaults to BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Used to select the subset of entity information to return.
    Defaults to BASIC.

    Values:
      ENTITY_VIEW_UNSPECIFIED: The API will default to the BASIC view.
      BASIC: Minimal view that does not include the schema.
      SCHEMA: Include basic information and schema.
      FULL: Include everything. Currently, this is the same as the SCHEMA
        view.
    """
        ENTITY_VIEW_UNSPECIFIED = 0
        BASIC = 1
        SCHEMA = 2
        FULL = 3
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)