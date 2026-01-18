from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsGetRequest(_messages.Message):
    """A VisionProjectsLocationsProductsGetRequest object.

  Fields:
    name: Required. Resource name of the Product to get. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`
  """
    name = _messages.StringField(1, required=True)