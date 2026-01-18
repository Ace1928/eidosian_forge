from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecParameterSpec(_messages.Message):
    """Represents a single parameter to optimize.

  Enums:
    ScaleTypeValueValuesEnum: How the parameter should be scaled. Leave unset
      for `CATEGORICAL` parameters.

  Fields:
    categoricalValueSpec: The value spec for a 'CATEGORICAL' parameter.
    conditionalParameterSpecs: A conditional parameter node is active if the
      parameter's value matches the conditional node's parent_value_condition.
      If two items in conditional_parameter_specs have the same name, they
      must have disjoint parent_value_condition.
    discreteValueSpec: The value spec for a 'DISCRETE' parameter.
    doubleValueSpec: The value spec for a 'DOUBLE' parameter.
    integerValueSpec: The value spec for an 'INTEGER' parameter.
    parameterId: Required. The ID of the parameter. Must not contain
      whitespaces and must be unique amongst all ParameterSpecs.
    scaleType: How the parameter should be scaled. Leave unset for
      `CATEGORICAL` parameters.
  """

    class ScaleTypeValueValuesEnum(_messages.Enum):
        """How the parameter should be scaled. Leave unset for `CATEGORICAL`
    parameters.

    Values:
      SCALE_TYPE_UNSPECIFIED: By default, no scaling is applied.
      UNIT_LINEAR_SCALE: Scales the feasible space to (0, 1) linearly.
      UNIT_LOG_SCALE: Scales the feasible space logarithmically to (0, 1). The
        entire feasible space must be strictly positive.
      UNIT_REVERSE_LOG_SCALE: Scales the feasible space "reverse"
        logarithmically to (0, 1). The result is that values close to the top
        of the feasible space are spread out more than points near the bottom.
        The entire feasible space must be strictly positive.
    """
        SCALE_TYPE_UNSPECIFIED = 0
        UNIT_LINEAR_SCALE = 1
        UNIT_LOG_SCALE = 2
        UNIT_REVERSE_LOG_SCALE = 3
    categoricalValueSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecCategoricalValueSpec', 1)
    conditionalParameterSpecs = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpec', 2, repeated=True)
    discreteValueSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecDiscreteValueSpec', 3)
    doubleValueSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecDoubleValueSpec', 4)
    integerValueSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecIntegerValueSpec', 5)
    parameterId = _messages.StringField(6)
    scaleType = _messages.EnumField('ScaleTypeValueValuesEnum', 7)