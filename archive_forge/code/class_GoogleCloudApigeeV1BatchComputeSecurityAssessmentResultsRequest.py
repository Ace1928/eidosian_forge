from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest(_messages.Message):
    """Request for BatchComputeSecurityAssessmentResults.

  Fields:
    include: Include only these resources.
    includeAllResources: Include all resources under the scope.
    pageSize: Optional. The maximum number of results to return. The service
      may return fewer than this value. If unspecified, at most 50 results
      will be returned.
    pageToken: Optional. A page token, received from a previous
      `BatchComputeSecurityAssessmentResults` call. Provide this to retrieve
      the subsequent page.
    profile: Required. Name of the profile that is used for computation.
    scope: Required. Scope of the resources for the computation. For Apigee,
      the environment is the scope of the resources.
  """
    include = _messages.MessageField('GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequestResourceArray', 1)
    includeAllResources = _messages.MessageField('GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequestIncludeAll', 2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    profile = _messages.StringField(5)
    scope = _messages.StringField(6)