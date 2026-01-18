from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetEffectiveIamPoliciesBatchGetRequest(_messages.Message):
    """A CloudassetEffectiveIamPoliciesBatchGetRequest object.

  Fields:
    names: Required. The names refer to the [full_resource_names]
      (https://cloud.google.com/asset-inventory/docs/resource-name-format) of
      the asset types [supported by search
      APIs](https://cloud.google.com/asset-inventory/docs/supported-asset-
      types). A maximum of 20 resources' effective policies can be retrieved
      in a batch.
    scope: Required. Only IAM policies on or below the scope will be returned.
      This can only be an organization number (such as "organizations/123"), a
      folder number (such as "folders/123"), a project ID (such as
      "projects/my-project-id"), or a project number (such as
      "projects/12345"). To know how to get organization ID, visit [here
      ](https://cloud.google.com/resource-manager/docs/creating-managing-
      organization#retrieving_your_organization_id). To know how to get folder
      or project ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      folders#viewing_or_listing_folders_and_projects).
  """
    names = _messages.StringField(1, repeated=True)
    scope = _messages.StringField(2, required=True)