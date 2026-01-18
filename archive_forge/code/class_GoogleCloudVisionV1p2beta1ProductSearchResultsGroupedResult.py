from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1ProductSearchResultsGroupedResult(_messages.Message):
    """Information about the products similar to a single product in a query
  image.

  Fields:
    boundingPoly: The bounding polygon around the product detected in the
      query image.
    objectAnnotations: List of generic predictions for the object in the
      bounding box.
    results: List of results, one for each product match.
  """
    boundingPoly = _messages.MessageField('GoogleCloudVisionV1p2beta1BoundingPoly', 1)
    objectAnnotations = _messages.MessageField('GoogleCloudVisionV1p2beta1ProductSearchResultsObjectAnnotation', 2, repeated=True)
    results = _messages.MessageField('GoogleCloudVisionV1p2beta1ProductSearchResultsResult', 3, repeated=True)