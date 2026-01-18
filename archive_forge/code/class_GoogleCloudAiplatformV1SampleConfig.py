from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SampleConfig(_messages.Message):
    """Active learning data sampling config. For every active learning labeling
  iteration, it will select a batch of data based on the sampling strategy.

  Enums:
    SampleStrategyValueValuesEnum: Field to choose sampling strategy. Sampling
      strategy will decide which data should be selected for human labeling in
      every batch.

  Fields:
    followingBatchSamplePercentage: The percentage of data needed to be
      labeled in each following batch (except the first batch).
    initialBatchSamplePercentage: The percentage of data needed to be labeled
      in the first batch.
    sampleStrategy: Field to choose sampling strategy. Sampling strategy will
      decide which data should be selected for human labeling in every batch.
  """

    class SampleStrategyValueValuesEnum(_messages.Enum):
        """Field to choose sampling strategy. Sampling strategy will decide which
    data should be selected for human labeling in every batch.

    Values:
      SAMPLE_STRATEGY_UNSPECIFIED: Default will be treated as UNCERTAINTY.
      UNCERTAINTY: Sample the most uncertain data to label.
    """
        SAMPLE_STRATEGY_UNSPECIFIED = 0
        UNCERTAINTY = 1
    followingBatchSamplePercentage = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    initialBatchSamplePercentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    sampleStrategy = _messages.EnumField('SampleStrategyValueValuesEnum', 3)