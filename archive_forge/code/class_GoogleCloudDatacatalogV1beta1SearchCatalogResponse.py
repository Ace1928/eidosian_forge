from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1SearchCatalogResponse(_messages.Message):
    """Response message for SearchCatalog.

  Fields:
    nextPageToken: The token that can be used to retrieve the next page of
      results.
    results: Search results.
    totalSize: The approximate total number of entries matched by the query.
    unreachable: Unreachable locations. Search result does not include data
      from those locations. Users can get additional information on the error
      by repeating the search request with a more restrictive parameter --
      setting the value for
      `SearchDataCatalogRequest.scope.restricted_locations`.
  """
    nextPageToken = _messages.StringField(1)
    results = _messages.MessageField('GoogleCloudDatacatalogV1beta1SearchCatalogResult', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    unreachable = _messages.StringField(4, repeated=True)