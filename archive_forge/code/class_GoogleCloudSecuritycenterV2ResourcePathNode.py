from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2ResourcePathNode(_messages.Message):
    """A node within the resource path. Each node represents a resource within
  the resource hierarchy.

  Enums:
    NodeTypeValueValuesEnum: The type of resource this node represents.

  Fields:
    displayName: The display name of the resource this node represents.
    id: The ID of the resource this node represents.
    nodeType: The type of resource this node represents.
  """

    class NodeTypeValueValuesEnum(_messages.Enum):
        """The type of resource this node represents.

    Values:
      RESOURCE_PATH_NODE_TYPE_UNSPECIFIED: Node type is unspecified.
      GCP_ORGANIZATION: The node represents a GCP organization.
      GCP_FOLDER: The node represents a GCP folder.
      GCP_PROJECT: The node represents a GCP project.
      AWS_ORGANIZATION: The node represents an AWS organization.
      AWS_ORGANIZATIONAL_UNIT: The node represents an AWS organizational unit.
      AWS_ACCOUNT: The node represents an AWS account.
      AZURE_MANAGEMENT_GROUP: The node represents an Azure management group.
      AZURE_SUBSCRIPTION: The node represents an Azure subscription.
      AZURE_RESOURCE_GROUP: The node represents an Azure resource group.
    """
        RESOURCE_PATH_NODE_TYPE_UNSPECIFIED = 0
        GCP_ORGANIZATION = 1
        GCP_FOLDER = 2
        GCP_PROJECT = 3
        AWS_ORGANIZATION = 4
        AWS_ORGANIZATIONAL_UNIT = 5
        AWS_ACCOUNT = 6
        AZURE_MANAGEMENT_GROUP = 7
        AZURE_SUBSCRIPTION = 8
        AZURE_RESOURCE_GROUP = 9
    displayName = _messages.StringField(1)
    id = _messages.StringField(2)
    nodeType = _messages.EnumField('NodeTypeValueValuesEnum', 3)