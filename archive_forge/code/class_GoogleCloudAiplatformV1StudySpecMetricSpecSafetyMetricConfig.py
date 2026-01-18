from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecMetricSpecSafetyMetricConfig(_messages.Message):
    """Used in safe optimization to specify threshold levels and risk
  tolerance.

  Fields:
    desiredMinSafeTrialsFraction: Desired minimum fraction of safe trials
      (over total number of trials) that should be targeted by the algorithm
      at any time during the study (best effort). This should be between 0.0
      and 1.0 and a value of 0.0 means that there is no minimum and an
      algorithm proceeds without targeting any specific fraction. A value of
      1.0 means that the algorithm attempts to only Suggest safe Trials.
    safetyThreshold: Safety threshold (boundary value between safe and
      unsafe). NOTE that if you leave SafetyMetricConfig unset, a default
      value of 0 will be used.
  """
    desiredMinSafeTrialsFraction = _messages.FloatField(1)
    safetyThreshold = _messages.FloatField(2)