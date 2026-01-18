from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsListRequest(_messages.Message):
    """A AppengineAppsServicesVersionsListRequest object.

  Enums:
    ViewValueValuesEnum: Controls the set of fields returned in the List
      response.

  Fields:
    pageSize: Maximum results to return per page.
    pageToken: Continuation token for fetching the next page of results.
    parent: Name of the parent Service resource. Example:
      apps/myapp/services/default.
    view: Controls the set of fields returned in the List response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Controls the set of fields returned in the List response.

    Values:
      BASIC: Basic version information including scaling and inbound services,
        but not detailed deployment information.
      FULL: The information from BASIC, plus detailed information about the
        deployment. This format is required when creating resources, but is
        not returned in Get or List by default.
    """
        BASIC = 0
        FULL = 1
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)