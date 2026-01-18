from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeOrgPolicyGovernedResourcesRequest(_messages.Message):
    """A CloudassetAnalyzeOrgPolicyGovernedResourcesRequest object.

  Fields:
    constraint: Required. The name of the constraint to analyze governed
      resources for. The analysis only contains analyzed organization policies
      for the provided constraint.
    filter: The expression to filter
      AnalyzeOrgPolicyGovernedResourcesResponse.governed_resources. Filtering
      is currently available for bare literal values and the following fields:
      * project * folders * consolidated_policy.rules.enforce When filtering
      by a specific field, the only supported operator is `=`. For example,
      filtering by project="projects/12345678" will return all the governed
      resources under "projects/12345678", including the project itself, if
      applicable.
    pageSize: The maximum number of items to return per page. If unspecified,
      AnalyzeOrgPolicyGovernedResourcesResponse.governed_resources will
      contain 100 items with a maximum of 200.
    pageToken: The pagination token to retrieve the next page.
    scope: Required. The organization to scope the request. Only organization
      policies within the scope will be analyzed. The output resources will
      also be limited to the ones governed by those in-scope organization
      policies. * organizations/{ORGANIZATION_NUMBER} (e.g.,
      "organizations/123456")
  """
    constraint = _messages.StringField(1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    scope = _messages.StringField(5, required=True)