from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecParameterSpecDiscreteValueSpec(_messages.Message):
    """Value specification for a parameter in `DISCRETE` type.

  Fields:
    defaultValue: A default value for a `DISCRETE` parameter that is assumed
      to be a relatively good starting point. Unset value signals that there
      is no offered starting point. It automatically rounds to the nearest
      feasible discrete point. Currently only supported by the Vertex AI
      Vizier service. Not supported by HyperparameterTuningJob or
      TrainingPipeline.
    values: Required. A list of possible values. The list should be in
      increasing order and at least 1e-10 apart. For instance, this parameter
      might have possible settings of 1.5, 2.5, and 4.0. This list should not
      contain more than 1,000 values.
  """
    defaultValue = _messages.FloatField(1)
    values = _messages.FloatField(2, repeated=True)