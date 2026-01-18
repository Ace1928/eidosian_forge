from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Measurement(_messages.Message):
    """A message representing a Measurement of a Trial. A Measurement contains
  the Metrics got by executing a Trial using suggested hyperparameter values.

  Fields:
    elapsedDuration: Output only. Time that the Trial has been running at the
      point of this Measurement.
    metrics: Output only. A list of metrics got by evaluating the objective
      functions using suggested Parameter values.
    stepCount: Output only. The number of steps the machine learning model has
      been trained for. Must be non-negative.
  """
    elapsedDuration = _messages.StringField(1)
    metrics = _messages.MessageField('GoogleCloudAiplatformV1MeasurementMetric', 2, repeated=True)
    stepCount = _messages.IntegerField(3)