from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsListRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsListRequest object.

  Fields:
    filter: Optional. A filter specifying constraints of a list operation. For
      example, `Domain.fqdn="mydomain.myorginization"`.
    orderBy: Optional. Specifies the ordering of results. See [Sorting
      order](https://cloud.google.com/apis/design/design_patterns#sorting_orde
      r) for more information.
    pageSize: Optional. The maximum number of items to return. If not
      specified, a default value of 1000 will be used. Regardless of the
      page_size value, the response may include a partial list. Callers should
      rely on a response's next_page_token to determine if there are
      additional results to list.
    pageToken: Optional. The `next_page_token` value returned from a previous
      ListDomainsRequest request, if any.
    parent: Required. The resource name of the domain location using the form:
      `projects/{project_id}/locations/global`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)