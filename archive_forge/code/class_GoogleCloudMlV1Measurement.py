from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Measurement(_messages.Message):
    """A message representing a measurement.

  Fields:
    elapsedTime: Output only. Time that the trial has been running at the
      point of this measurement.
    metrics: Provides a list of metrics that act as inputs into the objective
      function.
    stepCount: The number of steps a machine learning model has been trained
      for. Must be non-negative.
  """
    elapsedTime = _messages.StringField(1)
    metrics = _messages.MessageField('GoogleCloudMlV1MeasurementMetric', 2, repeated=True)
    stepCount = _messages.IntegerField(3)