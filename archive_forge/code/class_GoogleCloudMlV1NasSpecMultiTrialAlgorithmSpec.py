from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasSpecMultiTrialAlgorithmSpec(_messages.Message):
    """The spec of multi-trial Neural Architecture Search (NAS).

  Enums:
    MultiTrialAlgorithmValueValuesEnum: Optional. The multi-trial Neural
      Architecture Search (NAS) algorithm type. Defaults to
      `NAS_MULTI_TRIAL_ALGORITHM_REINFORCEMENT_LEARNING`.

  Fields:
    initialIgnoredModelCount: If non-zero, it specifies the number of first
      models whose rewards will be ignored.
    maxFailedNasTrials: Optional. It decides when a Neural Architecture Search
      (NAS) job should fail. Defaults to zero.
    maxNasTrials: Optional. How many Neural Architecture Search (NAS) trials
      should be attempted.
    maxParallelNasTrials: Required. The number of Neural Architecture Search
      (NAS) trials to run concurrently.
    multiTrialAlgorithm: Optional. The multi-trial Neural Architecture Search
      (NAS) algorithm type. Defaults to
      `NAS_MULTI_TRIAL_ALGORITHM_REINFORCEMENT_LEARNING`.
    nasTargetRewardMetric: Required. The TensorFlow summary tag that the
      controller tries to optimize. Its value needs to be consistent with the
      TensorFlow summary tag that is reported by trainer (customer provided
      dockers).
  """

    class MultiTrialAlgorithmValueValuesEnum(_messages.Enum):
        """Optional. The multi-trial Neural Architecture Search (NAS) algorithm
    type. Defaults to `NAS_MULTI_TRIAL_ALGORITHM_REINFORCEMENT_LEARNING`.

    Values:
      MULTI_TRIAL_ALGORITHM_UNSPECIFIED: <no description>
      REINFORCEMENT_LEARNING: The Reinforcement Learning Algorithm for Multi-
        trial Neural Architecture Search (NAS).
      GRID_SEARCH: The Grid Search Algorithm for Multi-trial Neural
        Architecture Search (NAS).
      REGULARIZED_EVOLUTION: The Regularized evolution Algorithm for Multi-
        trial Neural Architecture Search (NAS).
    """
        MULTI_TRIAL_ALGORITHM_UNSPECIFIED = 0
        REINFORCEMENT_LEARNING = 1
        GRID_SEARCH = 2
        REGULARIZED_EVOLUTION = 3
    initialIgnoredModelCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxFailedNasTrials = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    maxNasTrials = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maxParallelNasTrials = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    multiTrialAlgorithm = _messages.EnumField('MultiTrialAlgorithmValueValuesEnum', 5)
    nasTargetRewardMetric = _messages.StringField(6)