from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsDeleteRequest(_messages.Message):
    """A VisionProjectsLocationsProductsDeleteRequest object.

  Fields:
    name: Required. Resource name of product to delete. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`
  """
    name = _messages.StringField(1, required=True)