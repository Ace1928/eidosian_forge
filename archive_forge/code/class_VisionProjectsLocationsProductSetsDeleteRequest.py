from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsDeleteRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsDeleteRequest object.

  Fields:
    name: Required. Resource name of the ProductSet to delete. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/productSets/PRODUCT_SET_ID`
  """
    name = _messages.StringField(1, required=True)