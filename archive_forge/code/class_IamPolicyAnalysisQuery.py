from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicyAnalysisQuery(_messages.Message):
    """IAM policy analysis query message.

  Fields:
    accessSelector: Optional. Specifies roles or permissions for analysis.
      This is optional.
    conditionContext: Optional. The hypothetical context for IAM conditions
      evaluation.
    identitySelector: Optional. Specifies an identity for analysis.
    options: Optional. The query options.
    resourceSelector: Optional. Specifies a resource for analysis.
    scope: Required. The relative name of the root asset. Only resources and
      IAM policies within the scope will be analyzed. This can only be an
      organization number (such as "organizations/123"), a folder number (such
      as "folders/123"), a project ID (such as "projects/my-project-id"), or a
      project number (such as "projects/12345"). To know how to get
      organization ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      organization#retrieving_your_organization_id). To know how to get folder
      or project ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      folders#viewing_or_listing_folders_and_projects).
  """
    accessSelector = _messages.MessageField('AccessSelector', 1)
    conditionContext = _messages.MessageField('ConditionContext', 2)
    identitySelector = _messages.MessageField('IdentitySelector', 3)
    options = _messages.MessageField('Options', 4)
    resourceSelector = _messages.MessageField('ResourceSelector', 5)
    scope = _messages.StringField(6)