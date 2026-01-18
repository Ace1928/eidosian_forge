from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1TrainProcessorVersionRequestFoundationModelTuningOptions(_messages.Message):
    """Options to control foundation model tuning of the processor.

  Fields:
    learningRateMultiplier: Optional. The multiplier to apply to the
      recommended learning rate. Valid values are between 0.1 and 10. If not
      provided, recommended learning rate will be used.
    trainSteps: Optional. The number of steps to run for model tuning. Valid
      values are between 1 and 400. If not provided, recommended steps will be
      used.
  """
    learningRateMultiplier = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    trainSteps = _messages.IntegerField(2, variant=_messages.Variant.INT32)