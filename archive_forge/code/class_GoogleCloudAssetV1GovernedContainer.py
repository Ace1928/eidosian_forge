from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1GovernedContainer(_messages.Message):
    """The organization/folder/project resource governed by organization
  policies of AnalyzeOrgPolicyGovernedContainersRequest.constraint.

  Fields:
    consolidatedPolicy: The consolidated organization policy for the analyzed
      resource. The consolidated organization policy is computed by merging
      and evaluating AnalyzeOrgPolicyGovernedContainersResponse.GovernedContai
      ner.policy_bundle. The evaluation will respect the organization policy
      [hierarchy rules](https://cloud.google.com/resource-
      manager/docs/organization-policy/understanding-hierarchy).
    effectiveTags: The effective tags on this resource.
    folders: The folder(s) that this resource belongs to, in the format of
      folders/{FOLDER_NUMBER}. This field is available when the resource
      belongs (directly or cascadingly) to one or more folders.
    fullResourceName: The [full resource name]
      (https://cloud.google.com/asset-inventory/docs/resource-name-format) of
      an organization/folder/project resource.
    organization: The organization that this resource belongs to, in the
      format of organizations/{ORGANIZATION_NUMBER}. This field is available
      when the resource belongs (directly or cascadingly) to an organization.
    parent: The [full resource name] (https://cloud.google.com/asset-
      inventory/docs/resource-name-format) of the parent of AnalyzeOrgPolicyGo
      vernedContainersResponse.GovernedContainer.full_resource_name.
    policyBundle: The ordered list of all organization policies from the Analy
      zeOrgPoliciesResponse.OrgPolicyResult.consolidated_policy.attached_resou
      rce. to the scope specified in the request. If the constraint is defined
      with default policy, it will also appear in the list.
    project: The project that this resource belongs to, in the format of
      projects/{PROJECT_NUMBER}. This field is available when the resource
      belongs to a project.
  """
    consolidatedPolicy = _messages.MessageField('AnalyzerOrgPolicy', 1)
    effectiveTags = _messages.MessageField('EffectiveTagDetails', 2, repeated=True)
    folders = _messages.StringField(3, repeated=True)
    fullResourceName = _messages.StringField(4)
    organization = _messages.StringField(5)
    parent = _messages.StringField(6)
    policyBundle = _messages.MessageField('AnalyzerOrgPolicy', 7, repeated=True)
    project = _messages.StringField(8)