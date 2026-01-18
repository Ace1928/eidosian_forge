from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1ProductSearchResults(_messages.Message):
    """Results for a product search request.

  Fields:
    indexTime: Timestamp of the index which provided these results. Products
      added to the product set and products removed from the product set after
      this time are not reflected in the current results.
    productGroupedResults: List of results grouped by products detected in the
      query image. Each entry corresponds to one bounding polygon in the query
      image, and contains the matching products specific to that region. There
      may be duplicate product matches in the union of all the per-product
      results.
    results: List of results, one for each product match.
  """
    indexTime = _messages.StringField(1)
    productGroupedResults = _messages.MessageField('GoogleCloudVisionV1p2beta1ProductSearchResultsGroupedResult', 2, repeated=True)
    results = _messages.MessageField('GoogleCloudVisionV1p2beta1ProductSearchResultsResult', 3, repeated=True)