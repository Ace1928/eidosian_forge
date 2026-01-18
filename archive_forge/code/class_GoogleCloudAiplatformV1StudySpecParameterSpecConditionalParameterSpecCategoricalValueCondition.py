from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecParameterSpecConditionalParameterSpecCategoricalValueCondition(_messages.Message):
    """Represents the spec to match categorical values from parent parameter.

  Fields:
    values: Required. Matches values of the parent parameter of 'CATEGORICAL'
      type. All values must exist in `categorical_value_spec` of parent
      parameter.
  """
    values = _messages.StringField(1, repeated=True)