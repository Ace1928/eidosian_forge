from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpec(_messages.Message):
    """The spec of multi-trial Neural Architecture Search (NAS).

  Enums:
    MultiTrialAlgorithmValueValuesEnum: The multi-trial Neural Architecture
      Search (NAS) algorithm type. Defaults to `REINFORCEMENT_LEARNING`.

  Fields:
    metric: Metric specs for the NAS job. Validation for this field is done at
      `multi_trial_algorithm_spec` field.
    multiTrialAlgorithm: The multi-trial Neural Architecture Search (NAS)
      algorithm type. Defaults to `REINFORCEMENT_LEARNING`.
    searchTrialSpec: Required. Spec for search trials.
    trainTrialSpec: Spec for train trials. Top N
      [TrainTrialSpec.max_parallel_trial_count] search trials will be trained
      for every M [TrainTrialSpec.frequency] trials searched.
  """

    class MultiTrialAlgorithmValueValuesEnum(_messages.Enum):
        """The multi-trial Neural Architecture Search (NAS) algorithm type.
    Defaults to `REINFORCEMENT_LEARNING`.

    Values:
      MULTI_TRIAL_ALGORITHM_UNSPECIFIED: Defaults to `REINFORCEMENT_LEARNING`.
      REINFORCEMENT_LEARNING: The Reinforcement Learning Algorithm for Multi-
        trial Neural Architecture Search (NAS).
      GRID_SEARCH: The Grid Search Algorithm for Multi-trial Neural
        Architecture Search (NAS).
    """
        MULTI_TRIAL_ALGORITHM_UNSPECIFIED = 0
        REINFORCEMENT_LEARNING = 1
        GRID_SEARCH = 2
    metric = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpecMetricSpec', 1)
    multiTrialAlgorithm = _messages.EnumField('MultiTrialAlgorithmValueValuesEnum', 2)
    searchTrialSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpecSearchTrialSpec', 3)
    trainTrialSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpecTrainTrialSpec', 4)