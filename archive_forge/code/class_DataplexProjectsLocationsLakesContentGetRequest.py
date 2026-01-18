from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesContentGetRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesContentGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. Specify content view to make a partial
      request.

  Fields:
    name: Required. The resource name of the content: projects/{project_id}/lo
      cations/{location_id}/lakes/{lake_id}/content/{content_id}
    view: Optional. Specify content view to make a partial request.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Specify content view to make a partial request.

    Values:
      CONTENT_VIEW_UNSPECIFIED: Content view not specified. Defaults to BASIC.
        The API will default to the BASIC view.
      BASIC: Will not return the data_text field.
      FULL: Returns the complete proto.
    """
        CONTENT_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)