from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsCreateRequest(_messages.Message):
    """A VisionProjectsLocationsProductsCreateRequest object.

  Fields:
    parent: Required. The project in which the Product should be created.
      Format is `projects/PROJECT_ID/locations/LOC_ID`.
    product: A Product resource to be passed as the request body.
    productId: A user-supplied resource id for this Product. If set, the
      server will attempt to use this value as the resource id. If it is
      already in use, an error is returned with code ALREADY_EXISTS. Must be
      at most 128 characters long. It cannot contain the character `/`.
  """
    parent = _messages.StringField(1, required=True)
    product = _messages.MessageField('Product', 2)
    productId = _messages.StringField(3)