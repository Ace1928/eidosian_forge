from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsEkmConnectionsListRequest(_messages.Message):
    """A CloudkmsProjectsLocationsEkmConnectionsListRequest object.

  Fields:
    filter: Optional. Only include resources that match the filter in the
      response. For more information, see [Sorting and filtering list
      results](https://cloud.google.com/kms/docs/sorting-and-filtering).
    orderBy: Optional. Specify how the results should be sorted. If not
      specified, the results will be sorted in the default order. For more
      information, see [Sorting and filtering list
      results](https://cloud.google.com/kms/docs/sorting-and-filtering).
    pageSize: Optional. Optional limit on the number of EkmConnections to
      include in the response. Further EkmConnections can subsequently be
      obtained by including the ListEkmConnectionsResponse.next_page_token in
      a subsequent request. If unspecified, the server will pick an
      appropriate default.
    pageToken: Optional. Optional pagination token, returned earlier via
      ListEkmConnectionsResponse.next_page_token.
    parent: Required. The resource name of the location associated with the
      EkmConnections to list, in the format `projects/*/locations/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)