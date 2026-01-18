from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoveProductFromProductSetRequest(_messages.Message):
    """Request message for the `RemoveProductFromProductSet` method.

  Fields:
    product: Required. The resource name for the Product to be removed from
      this ProductSet. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID`
  """
    product = _messages.StringField(1)