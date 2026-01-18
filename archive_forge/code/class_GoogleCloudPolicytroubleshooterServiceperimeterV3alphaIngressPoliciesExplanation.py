from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIngressPoliciesExplanation(_messages.Message):
    """Explanation of ingress policies NextTAG: 5

  Enums:
    IngressPolicyEvalStatesValueListEntryValuesEnum:
    TopLevelAccessLevelsEvalStateValueValuesEnum: The overall evaluation state
      of the top level access levels

  Fields:
    ingressPolicyEvalStates: Details about the evaluation state of the ingress
      policy
    ingressPolicyExplanations: Explanations of ingress policies
    targetResource: The target resource to ingress to
    topLevelAccessLevelsEvalState: The overall evaluation state of the top
      level access levels
  """

    class IngressPolicyEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """IngressPolicyEvalStatesValueListEntryValuesEnum enum type.

    Values:
      INGRESS_POLICY_EVAL_STATE_UNSPECIFIED: Not used
      INGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER: The resources are
        in the same regular service perimeter
      INGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE: The resources are in the
        same bridge service perimeter
      INGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY: The request is granted by
        the ingress policy
      INGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY: The request is denied by the
        ingress policy
      INGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE: The ingress policy is
        applicable for the request
    """
        INGRESS_POLICY_EVAL_STATE_UNSPECIFIED = 0
        INGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER = 1
        INGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE = 2
        INGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY = 3
        INGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY = 4
        INGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE = 5

    class TopLevelAccessLevelsEvalStateValueValuesEnum(_messages.Enum):
        """The overall evaluation state of the top level access levels

    Values:
      TOP_LEVEL_ACCESS_LEVELS_EVAL_STATE_UNSPECIFIED: Not used
      NOT_APPLICABLE: The overall evaluation state of the top level access
        levels is not applicable
      GRANTED: The overall evaluation state of the top level access levels is
        granted
      DENIED: The overall evaluation state of the top level access levels is
        denied
    """
        TOP_LEVEL_ACCESS_LEVELS_EVAL_STATE_UNSPECIFIED = 0
        NOT_APPLICABLE = 1
        GRANTED = 2
        DENIED = 3
    ingressPolicyEvalStates = _messages.EnumField('IngressPolicyEvalStatesValueListEntryValuesEnum', 1, repeated=True)
    ingressPolicyExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIngressPolicyExplanation', 2, repeated=True)
    targetResource = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResource', 3)
    topLevelAccessLevelsEvalState = _messages.EnumField('TopLevelAccessLevelsEvalStateValueValuesEnum', 4)