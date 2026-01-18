from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstancesInventoriesListRequest(_messages.Message):
    """A OsconfigProjectsLocationsInstancesInventoriesListRequest object.

  Enums:
    ViewValueValuesEnum: Inventory view indicating what information should be
      included in the inventory resource. If unspecified, the default view is
      BASIC.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      a `Inventory` API resource to be included in the response.
    pageSize: The maximum number of results to return.
    pageToken: A pagination token returned from a previous call to
      `ListInventories` that indicates where this listing should continue
      from.
    parent: Required. The parent resource name. Format:
      `projects/{project}/locations/{location}/instances/-` For `{project}`,
      either `project-number` or `project-id` can be provided.
    view: Inventory view indicating what information should be included in the
      inventory resource. If unspecified, the default view is BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Inventory view indicating what information should be included in the
    inventory resource. If unspecified, the default view is BASIC.

    Values:
      INVENTORY_VIEW_UNSPECIFIED: The default value. The API defaults to the
        BASIC view.
      BASIC: Returns the basic inventory information that includes `os_info`.
      FULL: Returns all fields.
    """
        INVENTORY_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)