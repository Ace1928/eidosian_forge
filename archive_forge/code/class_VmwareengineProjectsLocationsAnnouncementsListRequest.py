from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsAnnouncementsListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsAnnouncementsListRequest object.

  Fields:
    filter: A filter expression that matches resources returned in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be `=`, `!=`, `>`, or `<`. For example, if you are filtering a list of
      announcement runs, you can exclude the ones named `example-announcement`
      by specifying `name != "example-announcement"`. You can also filter
      nested fields. To filter on multiple expressions, provide each separate
      expression within parentheses. For example: ``` (name = "example-
      announcement") (createTime > "2021-04-12T08:15:10.40Z") ``` By default,
      each expression is an `AND` expression. However, you can include `AND`
      and `OR` expressions explicitly. For example: ``` (name =
      "announcement-1") AND (createTime > "2021-04-12T08:15:10.40Z") OR (name
      = "announcement-2") ```
    orderBy: Sorts list results by a certain order. By default, returned
      results are ordered by `name` in ascending order. You can also sort
      results in descending order based on the `name` value using
      `orderBy="name desc"`. Currently, only ordering by `name` is supported.
    pageSize: The maximum number of announcements to return in one page. The
      service may return fewer than this value. The maximum value is coerced
      to 1000. The default value of this field is 500.
    pageToken: A page token, received from a previous `ListAnnouncements`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListAnnouncements` must match the call
      that provided the page token.
    parent: Required. The resource name of the location to be queried for
      announcements. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/us-west1-a`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)