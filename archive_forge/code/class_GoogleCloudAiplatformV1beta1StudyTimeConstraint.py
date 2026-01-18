from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudyTimeConstraint(_messages.Message):
    """Time-based Constraint for Study

  Fields:
    endTime: Compares the wallclock time to this time. Must use UTC timezone.
    maxDuration: Counts the wallclock time passed since the creation of this
      Study.
  """
    endTime = _messages.StringField(1)
    maxDuration = _messages.StringField(2)