from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersGetRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersGetRequest
  object.

  Fields:
    name: Required. The name of the order to retrieve.
  """
    name = _messages.StringField(1, required=True)