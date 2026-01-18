from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsBindingsListRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsBindingsListRequest object.

  Fields:
    filter: Optional. Lists MembershipBindings that match the filter
      expression, following the syntax outlined in https://google.aip.dev/160.
    pageSize: Optional. When requesting a 'page' of resources, `page_size`
      specifies number of resources to return. If unspecified or set to 0, all
      resources will be returned.
    pageToken: Optional. Token returned by previous call to
      `ListMembershipBindings` which specifies the position in the list from
      where to continue listing the resources.
    parent: Required. The parent Membership for which the MembershipBindings
      will be listed. Specified in the format
      `projects/*/locations/*/memberships/*`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)