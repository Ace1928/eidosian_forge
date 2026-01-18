from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsProductsListRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsProductsListRequest object.

  Fields:
    name: Required. The ProductSet resource for which to retrieve Products.
      Format is:
      `projects/PROJECT_ID/locations/LOC_ID/productSets/PRODUCT_SET_ID`
    pageSize: The maximum number of items to return. Default 10, maximum 100.
    pageToken: The next_page_token returned from a previous List request, if
      any.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)