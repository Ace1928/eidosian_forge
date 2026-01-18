from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ActiveLearningConfig(_messages.Message):
    """Parameters that configure the active learning pipeline. Active learning
  will label the data incrementally by several iterations. For every
  iteration, it will select a batch of data based on the sampling strategy.

  Fields:
    maxDataItemCount: Max number of human labeled DataItems.
    maxDataItemPercentage: Max percent of total DataItems for human labeling.
    sampleConfig: Active learning data sampling config. For every active
      learning labeling iteration, it will select a batch of data based on the
      sampling strategy.
    trainingConfig: CMLE training config. For every active learning labeling
      iteration, system will train a machine learning model on CMLE. The
      trained model will be used by data sampling algorithm to select
      DataItems.
  """
    maxDataItemCount = _messages.IntegerField(1)
    maxDataItemPercentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    sampleConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SampleConfig', 3)
    trainingConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1TrainingConfig', 4)