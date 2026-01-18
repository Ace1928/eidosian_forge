from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1AnalyzeOrgPolicyGovernedAssetsResponseGovernedIamPolicy(_messages.Message):
    """The IAM policies governed by the organization policies of the
  AnalyzeOrgPolicyGovernedAssetsRequest.constraint.

  Fields:
    assetType: The asset type of the AnalyzeOrgPolicyGovernedAssetsResponse.Go
      vernedIamPolicy.attached_resource. Example:
      `cloudresourcemanager.googleapis.com/Project` See [Cloud Asset Inventory
      Supported Asset Types](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types) for all supported asset types.
    attachedResource: The full resource name of the resource on which this IAM
      policy is set. Example: `//compute.googleapis.com/projects/my_project_12
      3/zones/zone1/instances/instance1`. See [Cloud Asset Inventory Resource
      Name Format](https://cloud.google.com/asset-inventory/docs/resource-
      name-format) for more information.
    folders: The folder(s) that this IAM policy belongs to, in the format of
      folders/{FOLDER_NUMBER}. This field is available when the IAM policy
      belongs (directly or cascadingly) to one or more folders.
    organization: The organization that this IAM policy belongs to, in the
      format of organizations/{ORGANIZATION_NUMBER}. This field is available
      when the IAM policy belongs (directly or cascadingly) to an
      organization.
    policy: The IAM policy directly set on the given resource.
    project: The project that this IAM policy belongs to, in the format of
      projects/{PROJECT_NUMBER}. This field is available when the IAM policy
      belongs to a project.
  """
    assetType = _messages.StringField(1)
    attachedResource = _messages.StringField(2)
    folders = _messages.StringField(3, repeated=True)
    organization = _messages.StringField(4)
    policy = _messages.MessageField('Policy', 5)
    project = _messages.StringField(6)