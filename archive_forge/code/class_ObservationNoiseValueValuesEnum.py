from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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