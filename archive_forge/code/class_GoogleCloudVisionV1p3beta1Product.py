from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1Product(_messages.Message):
    """A Product contains ReferenceImages.

  Fields:
    description: User-provided metadata to be stored with this product. Must
      be at most 4096 characters long.
    displayName: The user-provided name for this Product. Must not be empty.
      Must be at most 4096 characters long.
    name: The resource name of the product. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`. This field
      is ignored when creating a product.
    productCategory: Immutable. The category for the product identified by the
      reference image. This should be one of "homegoods-v2", "apparel-v2",
      "toys-v2", "packagedgoods-v1" or "general-v1". The legacy categories
      "homegoods", "apparel", and "toys" are still supported, but these should
      not be used for new products.
    productLabels: Key-value pairs that can be attached to a product. At query
      time, constraints can be specified based on the product_labels. Note
      that integer values can be provided as strings, e.g. "1199". Only
      strings with integer values can match a range-based restriction which is
      to be supported soon. Multiple values can be assigned to the same key.
      One product may have up to 500 product_labels. Notice that the total
      number of distinct product_labels over all products in one ProductSet
      cannot exceed 1M, otherwise the product search pipeline will refuse to
      work for that ProductSet.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)
    productCategory = _messages.StringField(4)
    productLabels = _messages.MessageField('GoogleCloudVisionV1p3beta1ProductKeyValue', 5, repeated=True)