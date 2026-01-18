from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecMedianAutomatedStoppingSpec(_messages.Message):
    """The median automated stopping rule stops a pending Trial if the Trial's
  best objective_value is strictly below the median 'performance' of all
  completed Trials reported up to the Trial's last measurement. Currently,
  'performance' refers to the running average of the objective values reported
  by the Trial in each measurement.

  Fields:
    useElapsedDuration: True if median automated stopping rule applies on
      Measurement.elapsed_duration. It means that elapsed_duration field of
      latest measurement of current Trial is used to compute median objective
      value for each completed Trials.
  """
    useElapsedDuration = _messages.BooleanField(1)