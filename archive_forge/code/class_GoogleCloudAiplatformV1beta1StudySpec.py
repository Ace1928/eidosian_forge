from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpec(_messages.Message):
    """Represents specification of a Study.

  Enums:
    AlgorithmValueValuesEnum: The search algorithm specified for the Study.
    MeasurementSelectionTypeValueValuesEnum: Describe which measurement
      selection type will be used
    ObservationNoiseValueValuesEnum: The observation noise level of the study.
      Currently only supported by the Vertex AI Vizier service. Not supported
      by HyperparameterTuningJob or TrainingPipeline.

  Fields:
    algorithm: The search algorithm specified for the Study.
    convexAutomatedStoppingSpec: The automated early stopping spec using
      convex stopping rule.
    convexStopConfig: Deprecated. The automated early stopping using convex
      stopping rule.
    decayCurveStoppingSpec: The automated early stopping spec using decay
      curve rule.
    measurementSelectionType: Describe which measurement selection type will
      be used
    medianAutomatedStoppingSpec: The automated early stopping spec using
      median rule.
    metrics: Required. Metric specs for the Study.
    observationNoise: The observation noise level of the study. Currently only
      supported by the Vertex AI Vizier service. Not supported by
      HyperparameterTuningJob or TrainingPipeline.
    parameters: Required. The set of parameters to tune.
    studyStoppingConfig: Conditions for automated stopping of a Study. Enable
      automated stopping by configuring at least one condition.
    transferLearningConfig: The configuration info/options for transfer
      learning. Currently supported for Vertex AI Vizier service, not
      HyperParameterTuningJob
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """The search algorithm specified for the Study.

    Values:
      ALGORITHM_UNSPECIFIED: The default algorithm used by Vertex AI for
        [hyperparameter tuning](https://cloud.google.com/vertex-
        ai/docs/training/hyperparameter-tuning-overview) and [Vertex AI
        Vizier](https://cloud.google.com/vertex-ai/docs/vizier).
      GRID_SEARCH: Simple grid search within the feasible space. To use grid
        search, all parameters must be `INTEGER`, `CATEGORICAL`, or
        `DISCRETE`.
      RANDOM_SEARCH: Simple random search within the feasible space.
    """
        ALGORITHM_UNSPECIFIED = 0
        GRID_SEARCH = 1
        RANDOM_SEARCH = 2

    class MeasurementSelectionTypeValueValuesEnum(_messages.Enum):
        """Describe which measurement selection type will be used

    Values:
      MEASUREMENT_SELECTION_TYPE_UNSPECIFIED: Will be treated as
        LAST_MEASUREMENT.
      LAST_MEASUREMENT: Use the last measurement reported.
      BEST_MEASUREMENT: Use the best measurement reported.
    """
        MEASUREMENT_SELECTION_TYPE_UNSPECIFIED = 0
        LAST_MEASUREMENT = 1
        BEST_MEASUREMENT = 2

    class ObservationNoiseValueValuesEnum(_messages.Enum):
        """The observation noise level of the study. Currently only supported by
    the Vertex AI Vizier service. Not supported by HyperparameterTuningJob or
    TrainingPipeline.

    Values:
      OBSERVATION_NOISE_UNSPECIFIED: The default noise level chosen by Vertex
        AI.
      LOW: Vertex AI assumes that the objective function is (nearly) perfectly
        reproducible, and will never repeat the same Trial parameters.
      HIGH: Vertex AI will estimate the amount of noise in metric evaluations,
        it may repeat the same Trial parameters more than once.
    """
        OBSERVATION_NOISE_UNSPECIFIED = 0
        LOW = 1
        HIGH = 2
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    convexAutomatedStoppingSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecConvexAutomatedStoppingSpec', 2)
    convexStopConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecConvexStopConfig', 3)
    decayCurveStoppingSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecDecayCurveAutomatedStoppingSpec', 4)
    measurementSelectionType = _messages.EnumField('MeasurementSelectionTypeValueValuesEnum', 5)
    medianAutomatedStoppingSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecMedianAutomatedStoppingSpec', 6)
    metrics = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecMetricSpec', 7, repeated=True)
    observationNoise = _messages.EnumField('ObservationNoiseValueValuesEnum', 8)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpec', 9, repeated=True)
    studyStoppingConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecStudyStoppingConfig', 10)
    transferLearningConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecTransferLearningConfig', 11)