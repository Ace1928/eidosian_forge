from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CutoverForecast(_messages.Message):
    """CutoverForecast holds information about future CutoverJobs of a
  MigratingVm.

  Fields:
    estimatedCutoverJobDuration: Output only. Estimation of the CutoverJob
      duration.
  """
    estimatedCutoverJobDuration = _messages.StringField(1)