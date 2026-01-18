from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsReferenceImagesCreateRequest(_messages.Message):
    """A VisionProjectsLocationsProductsReferenceImagesCreateRequest object.

  Fields:
    parent: Required. Resource name of the product in which to create the
      reference image. Format is
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`.
    referenceImage: A ReferenceImage resource to be passed as the request
      body.
    referenceImageId: A user-supplied resource id for the ReferenceImage to be
      added. If set, the server will attempt to use this value as the resource
      id. If it is already in use, an error is returned with code
      ALREADY_EXISTS. Must be at most 128 characters long. It cannot contain
      the character `/`.
  """
    parent = _messages.StringField(1, required=True)
    referenceImage = _messages.MessageField('ReferenceImage', 2)
    referenceImageId = _messages.StringField(3)