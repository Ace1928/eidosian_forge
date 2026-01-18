from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesGroupsMembersListRequest(_messages.Message):
    """A ServiceusageServicesGroupsMembersListRequest object.

  Enums:
    ViewValueValuesEnum: The view of the member state to use.

  Fields:
    pageSize: The maximum number of members to return. The service may return
      fewer than this value. If unspecified, at most 50 groups will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: A page token, received from a previous `ListGroupMembers` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListGroupMembers` must match the call that
      provided the page token.
    parent: Required. The parent group state that exposes the members.
    view: The view of the member state to use.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view of the member state to use.

    Values:
      MEMBER_STATE_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
      MEMBER_STATE_VIEW_BASIC: The basic view includes only the member
        metadata, it does not include information about the current state of
        the group. This is the default value.
      MEMBER_STATE_VIEW_FULL: Include everything.
    """
        MEMBER_STATE_VIEW_UNSPECIFIED = 0
        MEMBER_STATE_VIEW_BASIC = 1
        MEMBER_STATE_VIEW_FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)