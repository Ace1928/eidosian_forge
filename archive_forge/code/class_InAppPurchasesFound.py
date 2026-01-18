from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InAppPurchasesFound(_messages.Message):
    """Additional details of in-app purchases encountered during the crawl.

  Fields:
    inAppPurchasesFlowsExplored: The total number of in-app purchases flows
      explored: how many times the robo tries to buy a SKU.
    inAppPurchasesFlowsStarted: The total number of in-app purchases flows
      started.
  """
    inAppPurchasesFlowsExplored = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    inAppPurchasesFlowsStarted = _messages.IntegerField(2, variant=_messages.Variant.INT32)