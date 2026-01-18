from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootScoreBasedRoutingConfigRule(_messages.Message):
    """A LearningGenaiRootScoreBasedRoutingConfigRule object.

  Fields:
    equalOrGreaterThan: NOTE: Hardest examples have smaller values in their
      routing scores.
    lessThan: A LearningGenaiRootScore attribute.
    modelConfigId: This model_config_id points to ModelConfig::id which allows
      us to find the ModelConfig to route to. This is part of the banks
      specified in the ModelBankConfig.
  """
    equalOrGreaterThan = _messages.MessageField('LearningGenaiRootScore', 1)
    lessThan = _messages.MessageField('LearningGenaiRootScore', 2)
    modelConfigId = _messages.StringField(3)