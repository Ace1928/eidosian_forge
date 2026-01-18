from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsPatchRequest(_messages.Message):
    """A VisionProjectsLocationsProductsPatchRequest object.

  Fields:
    name: The resource name of the product. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`. This field
      is ignored when creating a product.
    product: A Product resource to be passed as the request body.
    updateMask: The FieldMask that specifies which fields to update. If
      update_mask isn't specified, all mutable fields are to be updated. Valid
      mask paths include `product_labels`, `display_name`, and `description`.
  """
    name = _messages.StringField(1, required=True)
    product = _messages.MessageField('Product', 2)
    updateMask = _messages.StringField(3)