from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpecDiscreteValueCondition(_messages.Message):
    """Represents the spec to match discrete values from parent parameter.

  Fields:
    values: Required. Matches values of the parent parameter of 'DISCRETE'
      type. All values must exist in `discrete_value_spec` of parent
      parameter. The Epsilon of the value matching is 1e-10.
  """
    values = _messages.FloatField(1, repeated=True)