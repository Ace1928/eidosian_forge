from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsListRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsListRequest object.

  Fields:
    filter: Optional. Only include resources that match the filter in the
      response.
    orderBy: Optional. Specify how the results should be sorted.
    pageSize: Optional. Limit on the number of CaPools to include in the
      response. Further CaPools can subsequently be obtained by including the
      ListCaPoolsResponse.next_page_token in a subsequent request. If
      unspecified, the server will pick an appropriate default.
    pageToken: Optional. Pagination token, returned earlier via
      ListCaPoolsResponse.next_page_token.
    parent: Required. The resource name of the location associated with the
      CaPools, in the format `projects/*/locations/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)