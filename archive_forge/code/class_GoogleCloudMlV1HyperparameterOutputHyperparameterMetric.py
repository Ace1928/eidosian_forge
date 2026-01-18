from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1HyperparameterOutputHyperparameterMetric(_messages.Message):
    """An observed value of a metric.

  Fields:
    objectiveValue: The objective value at this training step.
    trainingStep: The global training step for this metric.
  """
    objectiveValue = _messages.FloatField(1)
    trainingStep = _messages.IntegerField(2)