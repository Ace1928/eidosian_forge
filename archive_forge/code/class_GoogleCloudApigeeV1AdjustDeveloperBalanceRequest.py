from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AdjustDeveloperBalanceRequest(_messages.Message):
    """Request for AdjustDeveloperBalance.

  Fields:
    adjustment: * A positive value of `adjustment` means that that the API
      provider wants to adjust the balance for an under-charged developer i.e.
      the balance of the developer will decrease. * A negative value of
      `adjustment` means that that the API provider wants to adjust the
      balance for an over-charged developer i.e. the balance of the developer
      will increase.
  """
    adjustment = _messages.MessageField('GoogleTypeMoney', 1)