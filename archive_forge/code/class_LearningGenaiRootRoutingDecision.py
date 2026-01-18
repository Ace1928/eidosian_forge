from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRoutingDecision(_messages.Message):
    """Holds the final routing decision, by storing the model_config_id. And
  individual scores each model got.

  Fields:
    metadata: A LearningGenaiRootRoutingDecisionMetadata attribute.
    modelConfigId: The selected model to route traffic to.
  """
    metadata = _messages.MessageField('LearningGenaiRootRoutingDecisionMetadata', 1)
    modelConfigId = _messages.StringField(2)