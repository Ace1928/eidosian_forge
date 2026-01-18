from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsReferenceImagesListRequest(_messages.Message):
    """A VisionProjectsLocationsProductsReferenceImagesListRequest object.

  Fields:
    pageSize: The maximum number of items to return. Default 10, maximum 100.
    pageToken: A token identifying a page of results to be returned. This is
      the value of `nextPageToken` returned in a previous reference image list
      request. Defaults to the first page if not specified.
    parent: Required. Resource name of the product containing the reference
      images. Format is
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)