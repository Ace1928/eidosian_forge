from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecDiscreteValueSpec(_messages.Message):
    """A GoogleCloudMlV1StudyConfigParameterSpecDiscreteValueSpec object.

  Fields:
    values: Must be specified if type is `DISCRETE`. A list of feasible
      points. The list should be in strictly increasing order. For instance,
      this parameter might have possible settings of 1.5, 2.5, and 4.0. This
      list should not contain more than 1,000 values.
  """
    values = _messages.FloatField(1, repeated=True)