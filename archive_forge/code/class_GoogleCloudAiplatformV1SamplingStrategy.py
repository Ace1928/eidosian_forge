from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SamplingStrategy(_messages.Message):
    """Sampling Strategy for logging, can be for both training and prediction
  dataset.

  Fields:
    randomSampleConfig: Random sample config. Will support more sampling
      strategies later.
  """
    randomSampleConfig = _messages.MessageField('GoogleCloudAiplatformV1SamplingStrategyRandomSampleConfig', 1)