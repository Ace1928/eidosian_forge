from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1AutomatedStoppingConfigDecayCurveAutomatedStoppingConfig(_messages.Message):
    """A
  GoogleCloudMlV1AutomatedStoppingConfigDecayCurveAutomatedStoppingConfig
  object.

  Fields:
    useElapsedTime: If true, measurement.elapsed_time is used as the x-axis of
      each Trials Decay Curve. Otherwise, Measurement.steps will be used as
      the x-axis.
  """
    useElapsedTime = _messages.BooleanField(1)