from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1Resource(_messages.Message):
    """Information related to the Google Cloud resource.

  Enums:
    CloudProviderValueValuesEnum: Indicates which cloud provider the resource
      resides in.

  Fields:
    awsMetadata: The AWS metadata associated with the finding.
    cloudProvider: Indicates which cloud provider the resource resides in.
    displayName: The human readable name of the resource.
    folders: Output only. Contains a Folder message for each folder in the
      assets ancestry. The first folder is the deepest nested folder, and the
      last folder is the folder directly under the Organization.
    location: The region or location of the service (if applicable).
    name: The full resource name of the resource. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    organization: Indicates which organization or tenant in the cloud provider
      the finding applies to.
    parent: The full resource name of resource's parent.
    parentDisplayName: The human readable name of resource's parent.
    project: The full resource name of project that the resource belongs to.
    projectDisplayName: The project ID that the resource belongs to.
    resourcePath: Provides the path to the resource within the resource
      hierarchy.
    resourcePathString: A string representation of the resource path. For GCP,
      it has the format of: organizations/{organization_id}/folders/{folder_id
      }/folders/{folder_id}/projects/{project_id} where there can be any
      number of folders. For AWS, it has the format of: org/{organization_id}/
      ou/{organizational_unit_id}/ou/{organizational_unit_id}/account/{account
      _id} where there can be any number of organizational units. For Azure,
      it has the format of: mg/{management_group_id}/mg/{management_group_id}/
      subscription/{subscription_id}/rg/{resource_group_name} where there can
      be any number of management groups.
    service: The parent service or product from which the resource is
      provided, for example, GKE or SNS.
    type: The full resource type of the resource.
  """

    class CloudProviderValueValuesEnum(_messages.Enum):
        """Indicates which cloud provider the resource resides in.

    Values:
      CLOUD_PROVIDER_UNSPECIFIED: The cloud provider is unspecified.
      GOOGLE_CLOUD_PLATFORM: The cloud provider is Google Cloud Platform.
      AMAZON_WEB_SERVICES: The cloud provider is Amazon Web Services.
      MICROSOFT_AZURE: The cloud provider is Microsoft Azure.
    """
        CLOUD_PROVIDER_UNSPECIFIED = 0
        GOOGLE_CLOUD_PLATFORM = 1
        AMAZON_WEB_SERVICES = 2
        MICROSOFT_AZURE = 3
    awsMetadata = _messages.MessageField('AwsMetadata', 1)
    cloudProvider = _messages.EnumField('CloudProviderValueValuesEnum', 2)
    displayName = _messages.StringField(3)
    folders = _messages.MessageField('Folder', 4, repeated=True)
    location = _messages.StringField(5)
    name = _messages.StringField(6)
    organization = _messages.StringField(7)
    parent = _messages.StringField(8)
    parentDisplayName = _messages.StringField(9)
    project = _messages.StringField(10)
    projectDisplayName = _messages.StringField(11)
    resourcePath = _messages.MessageField('ResourcePath', 12)
    resourcePathString = _messages.StringField(13)
    service = _messages.StringField(14)
    type = _messages.StringField(15)