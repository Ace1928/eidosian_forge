from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SerialPipeline(_messages.Message):
    """SerialPipeline defines a sequential set of stages for a
  `DeliveryPipeline`.

  Fields:
    stages: Each stage specifies configuration for a `Target`. The ordering of
      this list defines the promotion flow.
  """
    stages = _messages.MessageField('Stage', 1, repeated=True)