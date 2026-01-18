from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsAccountsDeleteRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsAccountsDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the account to delete.
  """
    name = _messages.StringField(1, required=True)