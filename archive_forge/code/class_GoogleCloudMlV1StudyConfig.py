from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfig(_messages.Message):
    """Represents configuration of a study.

  Enums:
    AlgorithmValueValuesEnum: The search algorithm specified for the study.

  Fields:
    algorithm: The search algorithm specified for the study.
    automatedStoppingConfig: Configuration for automated stopping of
      unpromising Trials.
    metrics: Metric specs for the study.
    parameters: Required. The set of parameters to tune.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """The search algorithm specified for the study.

    Values:
      ALGORITHM_UNSPECIFIED: The default algorithm used by the Cloud AI
        Platform Vizier service.
      GAUSSIAN_PROCESS_BANDIT: Gaussian Process Bandit.
      GRID_SEARCH: Simple grid search within the feasible space. To use grid
        search, all parameters must be `INTEGER`, `CATEGORICAL`, or
        `DISCRETE`.
      RANDOM_SEARCH: Simple random search within the feasible space.
    """
        ALGORITHM_UNSPECIFIED = 0
        GAUSSIAN_PROCESS_BANDIT = 1
        GRID_SEARCH = 2
        RANDOM_SEARCH = 3
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    automatedStoppingConfig = _messages.MessageField('GoogleCloudMlV1AutomatedStoppingConfig', 2)
    metrics = _messages.MessageField('GoogleCloudMlV1StudyConfigMetricSpec', 3, repeated=True)
    parameters = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpec', 4, repeated=True)