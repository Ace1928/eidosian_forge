from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MarketplacesolutionsProjectsLocationsPowerInstancesResetRequest(_messages.Message):
    """A MarketplacesolutionsProjectsLocationsPowerInstancesResetRequest
  object.

  Fields:
    name: Required. Name of the resource.
    resetPowerInstanceRequest: A ResetPowerInstanceRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    resetPowerInstanceRequest = _messages.MessageField('ResetPowerInstanceRequest', 2)