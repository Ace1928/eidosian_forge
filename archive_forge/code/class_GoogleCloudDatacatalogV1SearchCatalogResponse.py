from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SearchCatalogResponse(_messages.Message):
    """Response message for SearchCatalog.

  Fields:
    nextPageToken: Pagination token that can be used in subsequent calls to
      retrieve the next page of results.
    results: Search results.
    totalSize: The approximate total number of entries matched by the query.
    unreachable: Unreachable locations. Search results don't include data from
      those locations. To get additional information on an error, repeat the
      search request and restrict it to specific locations by setting the
      `SearchCatalogRequest.scope.restricted_locations` parameter.
  """
    nextPageToken = _messages.StringField(1)
    results = _messages.MessageField('GoogleCloudDatacatalogV1SearchCatalogResult', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    unreachable = _messages.StringField(4, repeated=True)