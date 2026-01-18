from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecMatchingParentDiscreteValueSpec(_messages.Message):
    """Represents the spec to match discrete values from parent parameter.

  Fields:
    values: Matches values of the parent parameter with type 'DISCRETE'. All
      values must exist in `discrete_value_spec` of parent parameter.
  """
    values = _messages.FloatField(1, repeated=True)