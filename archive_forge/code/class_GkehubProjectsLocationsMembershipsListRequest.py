from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsListRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsListRequest object.

  Fields:
    filter: Optional. Lists Memberships that match the filter expression,
      following the syntax outlined in https://google.aip.dev/160. Examples: -
      Name is `bar` in project `foo-proj` and location `global`: name =
      "projects/foo-proj/locations/global/membership/bar" - Memberships that
      have a label called `foo`: labels.foo:* - Memberships that have a label
      called `foo` whose value is `bar`: labels.foo = bar - Memberships in the
      CREATING state: state = CREATING
    orderBy: Optional. One or more fields to compare and use to sort the
      output. See https://google.aip.dev/132#ordering.
    pageSize: Optional. When requesting a 'page' of resources, `page_size`
      specifies number of resources to return. If unspecified or set to 0, all
      resources will be returned.
    pageToken: Optional. Token returned by previous call to `ListMemberships`
      which specifies the position in the list from where to continue listing
      the resources.
    parent: Required. The parent (project and location) where the Memberships
      will be listed. Specified in the format `projects/*/locations/*`.
      `projects/*/locations/-` list memberships in all the regions.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)