from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecDoubleValueSpec(_messages.Message):
    """A GoogleCloudMlV1StudyConfigParameterSpecDoubleValueSpec object.

  Fields:
    maxValue: Must be specified if type is `DOUBLE`. Maximum value of the
      parameter.
    minValue: Must be specified if type is `DOUBLE`. Minimum value of the
      parameter.
  """
    maxValue = _messages.FloatField(1)
    minValue = _messages.FloatField(2)