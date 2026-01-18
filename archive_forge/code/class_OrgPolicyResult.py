from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgPolicyResult(_messages.Message):
    """The organization policy result to the query.

  Fields:
    consolidatedPolicy: The consolidated organization policy for the analyzed
      resource. The consolidated organization policy is computed by merging
      and evaluating AnalyzeOrgPoliciesResponse.policy_bundle. The evaluation
      will respect the organization policy [hierarchy
      rules](https://cloud.google.com/resource-manager/docs/organization-
      policy/understanding-hierarchy).
    folders: The folder(s) that this consolidated policy belongs to, in the
      format of folders/{FOLDER_NUMBER}. This field is available when the
      consolidated policy belongs (directly or cascadingly) to one or more
      folders.
    organization: The organization that this consolidated policy belongs to,
      in the format of organizations/{ORGANIZATION_NUMBER}. This field is
      available when the consolidated policy belongs (directly or cascadingly)
      to an organization.
    policyBundle: The ordered list of all organization policies from the Analy
      zeOrgPoliciesResponse.OrgPolicyResult.consolidated_policy.attached_resou
      rce. to the scope specified in the request. If the constraint is defined
      with default policy, it will also appear in the list.
    project: The project that this consolidated policy belongs to, in the
      format of projects/{PROJECT_NUMBER}. This field is available when the
      consolidated policy belongs to a project.
  """
    consolidatedPolicy = _messages.MessageField('AnalyzerOrgPolicy', 1)
    folders = _messages.StringField(2, repeated=True)
    organization = _messages.StringField(3)
    policyBundle = _messages.MessageField('AnalyzerOrgPolicy', 4, repeated=True)
    project = _messages.StringField(5)