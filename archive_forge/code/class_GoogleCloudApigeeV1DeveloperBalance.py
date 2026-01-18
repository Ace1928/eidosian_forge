from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeveloperBalance(_messages.Message):
    """Account balance for the developer.

  Fields:
    wallets: Output only. List of all wallets. Each individual wallet stores
      the account balance for a particular currency.
  """
    wallets = _messages.MessageField('GoogleCloudApigeeV1DeveloperBalanceWallet', 1, repeated=True)