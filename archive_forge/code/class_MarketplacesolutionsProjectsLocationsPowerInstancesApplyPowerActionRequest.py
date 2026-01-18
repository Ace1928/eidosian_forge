from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MarketplacesolutionsProjectsLocationsPowerInstancesApplyPowerActionRequest(_messages.Message):
    """A
  MarketplacesolutionsProjectsLocationsPowerInstancesApplyPowerActionRequest
  object.

  Fields:
    applyPowerInstancePowerActionRequest: A
      ApplyPowerInstancePowerActionRequest resource to be passed as the
      request body.
    name: Required. Name of the resource.
  """
    applyPowerInstancePowerActionRequest = _messages.MessageField('ApplyPowerInstancePowerActionRequest', 1)
    name = _messages.StringField(2, required=True)