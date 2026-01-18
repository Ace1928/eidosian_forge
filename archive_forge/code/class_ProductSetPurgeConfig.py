from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProductSetPurgeConfig(_messages.Message):
    """Config to control which ProductSet contains the Products to be deleted.

  Fields:
    productSetId: The ProductSet that contains the Products to delete. If a
      Product is a member of product_set_id in addition to other ProductSets,
      the Product will still be deleted.
  """
    productSetId = _messages.StringField(1)