from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIngressPolicyExplanation(_messages.Message):
    """Explanation of an ingress policy NextTAG: 8

  Enums:
    ApiOperationEvalStatesValueListEntryValuesEnum:
    IdentityTypeEvalStateValueValuesEnum: Details of the evaluation state of
      the identity type
    IngressPolicyEvalStateValueValuesEnum: The overall evaluation state of the
      ingress policy
    IngressSourceEvalStatesValueListEntryValuesEnum:
    ResourceEvalStatesValueListEntryValuesEnum:

  Fields:
    apiOperationEvalStates: Details of the evaluation states of api operations
    identityExplanations: Detailed explanation of the identities.
    identityTypeEvalState: Details of the evaluation state of the identity
      type
    ingressPolicyEvalState: The overall evaluation state of the ingress policy
    ingressSourceEvalStates: Details of the evaluation states of ingress
      sources
    resourceEvalStates: Details of the evaluation states of resources
  """

    class ApiOperationEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """ApiOperationEvalStatesValueListEntryValuesEnum enum type.

    Values:
      API_OPERATION_EVAL_STATE_UNSPECIFIED: Not used
      API_OPERATION_EVAL_STATE_MATCH: The request matches the api operation
      API_OPERATION_EVAL_STATE_NOT_MATCH: The request doesn't match the api
        operation
    """
        API_OPERATION_EVAL_STATE_UNSPECIFIED = 0
        API_OPERATION_EVAL_STATE_MATCH = 1
        API_OPERATION_EVAL_STATE_NOT_MATCH = 2

    class IdentityTypeEvalStateValueValuesEnum(_messages.Enum):
        """Details of the evaluation state of the identity type

    Values:
      IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED: Not used
      IDENTITY_TYPE_EVAL_STATE_GRANTED: The request type matches the identity
      IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED: The request type doesn't match the
        identity
      IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED: The identity type is not
        supported
    """
        IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED = 0
        IDENTITY_TYPE_EVAL_STATE_GRANTED = 1
        IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED = 2
        IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED = 3

    class IngressPolicyEvalStateValueValuesEnum(_messages.Enum):
        """The overall evaluation state of the ingress policy

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

    class IngressSourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """IngressSourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      INGRESS_SOURCE_EVAL_STATE_UNSPECIFIED: Not used
      INGRESS_SOURCE_EVAL_STATE_MATCH: The request matches the ingress source
      INGRESS_SOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the
        ingress source
    """
        INGRESS_SOURCE_EVAL_STATE_UNSPECIFIED = 0
        INGRESS_SOURCE_EVAL_STATE_MATCH = 1
        INGRESS_SOURCE_EVAL_STATE_NOT_MATCH = 2

    class ResourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """ResourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      RESOURCE_EVAL_STATE_UNSPECIFIED: Not used
      RESOURCE_EVAL_STATE_MATCH: The request matches the resource
      RESOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the resource
    """
        RESOURCE_EVAL_STATE_UNSPECIFIED = 0
        RESOURCE_EVAL_STATE_MATCH = 1
        RESOURCE_EVAL_STATE_NOT_MATCH = 2
    apiOperationEvalStates = _messages.EnumField('ApiOperationEvalStatesValueListEntryValuesEnum', 1, repeated=True)
    identityExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIdentityExplanation', 2, repeated=True)
    identityTypeEvalState = _messages.EnumField('IdentityTypeEvalStateValueValuesEnum', 3)
    ingressPolicyEvalState = _messages.EnumField('IngressPolicyEvalStateValueValuesEnum', 4)
    ingressSourceEvalStates = _messages.EnumField('IngressSourceEvalStatesValueListEntryValuesEnum', 5, repeated=True)
    resourceEvalStates = _messages.EnumField('ResourceEvalStatesValueListEntryValuesEnum', 6, repeated=True)