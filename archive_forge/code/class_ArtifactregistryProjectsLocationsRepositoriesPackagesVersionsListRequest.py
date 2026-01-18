from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsListRequest(_messages.Message):
    """A
  ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsListRequest
  object.

  Enums:
    ViewValueValuesEnum: The view that should be returned in the response.

  Fields:
    orderBy: Optional. The field to order the results by.
    pageSize: The maximum number of versions to return. Maximum page size is
      1,000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: The name of the parent resource whose versions will be listed.
    view: The view that should be returned in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view that should be returned in the response.

    Values:
      VERSION_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
      BASIC: Includes basic information about the version, but not any related
        tags.
      FULL: Include everything.
    """
        VERSION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)