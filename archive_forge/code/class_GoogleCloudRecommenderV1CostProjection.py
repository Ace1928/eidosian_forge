from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1CostProjection(_messages.Message):
    """Contains metadata about how much money a recommendation can save or
  incur.

  Fields:
    cost: An approximate projection on amount saved or amount incurred.
      Negative cost units indicate cost savings and positive cost units
      indicate increase. See google.type.Money documentation for
      positive/negative units. A user's permissions may affect whether the
      cost is computed using list prices or custom contract prices.
    costInLocalCurrency: The approximate cost savings in the billing account's
      local currency.
    duration: Duration for which this cost applies.
  """
    cost = _messages.MessageField('GoogleTypeMoney', 1)
    costInLocalCurrency = _messages.MessageField('GoogleTypeMoney', 2)
    duration = _messages.StringField(3)