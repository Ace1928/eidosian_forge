from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelatedAsset(_messages.Message):
    """An asset identifier in Google Cloud which contains its name, type and
  ancestors. An asset can be any resource in the Google Cloud [resource
  hierarchy](https://cloud.google.com/resource-manager/docs/cloud-platform-
  resource-hierarchy), a resource outside the Google Cloud resource hierarchy
  (such as Google Kubernetes Engine clusters and objects), or a policy (e.g.
  IAM policy). See [Supported asset types](https://cloud.google.com/asset-
  inventory/docs/supported-asset-types) for more information.

  Fields:
    ancestors: The ancestors of an asset in Google Cloud [resource
      hierarchy](https://cloud.google.com/resource-manager/docs/cloud-
      platform-resource-hierarchy), represented as a list of relative resource
      names. An ancestry path starts with the closest ancestor in the
      hierarchy and ends at root. Example: `["projects/123456789",
      "folders/5432", "organizations/1234"]`
    asset: The full name of the asset. Example: `//compute.googleapis.com/proj
      ects/my_project_123/zones/zone1/instances/instance1` See [Resource names
      ](https://cloud.google.com/apis/design/resource_names#full_resource_name
      ) for more information.
    assetType: The type of the asset. Example: `compute.googleapis.com/Disk`
      See [Supported asset types](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types) for more information.
    relationshipType: The unique identifier of the relationship type. Example:
      `INSTANCE_TO_INSTANCEGROUP`
  """
    ancestors = _messages.StringField(1, repeated=True)
    asset = _messages.StringField(2)
    assetType = _messages.StringField(3)
    relationshipType = _messages.StringField(4)