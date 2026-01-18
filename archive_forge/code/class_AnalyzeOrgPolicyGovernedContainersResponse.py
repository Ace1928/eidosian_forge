from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeOrgPolicyGovernedContainersResponse(_messages.Message):
    """The response message for
  AssetService.AnalyzeOrgPolicyGovernedContainers.

  Fields:
    constraint: The definition of the constraint in the request.
    governedContainers: The list of the analyzed governed containers.
    nextPageToken: The page token to fetch the next page for
      AnalyzeOrgPolicyGovernedContainersResponse.governed_containers.
  """
    constraint = _messages.MessageField('AnalyzerOrgPolicyConstraint', 1)
    governedContainers = _messages.MessageField('GoogleCloudAssetV1GovernedContainer', 2, repeated=True)
    nextPageToken = _messages.StringField(3)