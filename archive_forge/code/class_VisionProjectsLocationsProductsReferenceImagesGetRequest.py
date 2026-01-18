from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsReferenceImagesGetRequest(_messages.Message):
    """A VisionProjectsLocationsProductsReferenceImagesGetRequest object.

  Fields:
    name: Required. The resource name of the ReferenceImage to get. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID/referenceImage
      s/IMAGE_ID`.
  """
    name = _messages.StringField(1, required=True)