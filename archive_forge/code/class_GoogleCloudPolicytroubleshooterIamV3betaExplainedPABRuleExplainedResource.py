from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedPABRuleExplainedResource(_messages.Message):
    """Details about how a resource contributes to the explanation, with
  annotations to indicate how the resource contributes to the overall access
  state.

  Enums:
    RelevanceValueValuesEnum: The relevance of this resource to the overall
      access state.
    ResourceInclusionStateValueValuesEnum: Output only. Indicates whether the
      resource is the specified resource or includes the specified resource.

  Fields:
    relevance: The relevance of this resource to the overall access state.
    resource: The [full resource name](https://cloud.google.com/iam/docs/full-
      resource-names) that identifies the resource that is explained. This can
      only be a project, a folder, or an organization which is what a PAB rule
      accepts.
    resourceInclusionState: Output only. Indicates whether the resource is the
      specified resource or includes the specified resource.
  """

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this resource to the overall access state.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2

    class ResourceInclusionStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the resource is the specified resource
    or includes the specified resource.

    Values:
      RESOURCE_INCLUSION_STATE_UNSPECIFIED: An error occurred when checking
        whether the resource includes the specified resource.
      RESOURCE_INCLUSION_STATE_INCLUDED: The resource includes the specified
        resource.
      RESOURCE_INCLUSION_STATE_NOT_INCLUDED: The resource doesn't include the
        specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_INFO: The sender of the request does
        not have access to the relevant data to check whether the resource
        includes the specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED: The resource is of an
        unsupported type, such as non-CRM resources.
    """
        RESOURCE_INCLUSION_STATE_UNSPECIFIED = 0
        RESOURCE_INCLUSION_STATE_INCLUDED = 1
        RESOURCE_INCLUSION_STATE_NOT_INCLUDED = 2
        RESOURCE_INCLUSION_STATE_UNKNOWN_INFO = 3
        RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED = 4
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 1)
    resource = _messages.StringField(2)
    resourceInclusionState = _messages.EnumField('ResourceInclusionStateValueValuesEnum', 3)