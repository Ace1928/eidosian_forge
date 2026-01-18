from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1TrialParameter(_messages.Message):
    """A message representing a parameter to be tuned. Contains the name of the
  parameter and the suggested value to use for this trial.

  Fields:
    floatValue: Must be set if ParameterType is DOUBLE or DISCRETE.
    intValue: Must be set if ParameterType is INTEGER
    parameter: The name of the parameter.
    stringValue: Must be set if ParameterTypeis CATEGORICAL
  """
    floatValue = _messages.FloatField(1)
    intValue = _messages.IntegerField(2)
    parameter = _messages.StringField(3)
    stringValue = _messages.StringField(4)