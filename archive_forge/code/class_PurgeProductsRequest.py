from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PurgeProductsRequest(_messages.Message):
    """Request message for the `PurgeProducts` method.

  Fields:
    deleteOrphanProducts: If delete_orphan_products is true, all Products that
      are not in any ProductSet will be deleted.
    force: The default value is false. Override this value to true to actually
      perform the purge.
    productSetPurgeConfig: Specify which ProductSet contains the Products to
      be deleted.
  """
    deleteOrphanProducts = _messages.BooleanField(1)
    force = _messages.BooleanField(2)
    productSetPurgeConfig = _messages.MessageField('ProductSetPurgeConfig', 3)