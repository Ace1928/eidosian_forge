from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppconfigmanagerProjectsLocationsConfigsVersionsListRequest(_messages.Message):
    """A AppconfigmanagerProjectsLocationsConfigsVersionsListRequest object.

  Enums:
    ViewValueValuesEnum: Optional. View of the ConfigVersion. In the default
      BASIC view, only the metadata associated with the ConfigVersion will be
      returned.

  Fields:
    filter: Optional. Filtering results
    orderBy: Optional. Hint for how to order the results
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. Parent value for ListConfigsRequest
    view: Optional. View of the ConfigVersion. In the default BASIC view, only
      the metadata associated with the ConfigVersion will be returned.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. View of the ConfigVersion. In the default BASIC view, only
    the metadata associated with the ConfigVersion will be returned.

    Values:
      VIEW_UNSPECIFIED: The default / unset value. The API will default to the
        BASIC view for LIST calls & FULL for GET calls..
      BASIC: Include only the metadata for the resource. This is the default
        view.
      FULL: Include metadata & other relevant payload data as well. For a
        ConfigVersion this implies that the response will hold the user
        provided payload. For a ConfigVersionRender this implies that the
        response will hold the user provided payload along with the rendered
        payload data.
    """
        VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)