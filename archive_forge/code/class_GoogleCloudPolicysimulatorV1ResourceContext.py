from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1ResourceContext(_messages.Message):
    """ResourceContext provides the context we know about a resource. It is
  similar in concept to google.cloud.asset.v1.Resource, but focuses on the
  information specifically used by Simulator.

  Fields:
    ancestors: The ancestry path of the resource in Google Cloud [resource
      hierarchy](https://cloud.google.com/resource-manager/docs/cloud-
      platform-resource-hierarchy), represented as a list of relative resource
      names. An ancestry path starts with the closest ancestor in the
      hierarchy and ends at root. If the resource is a project, folder, or
      organization, the ancestry path starts from the resource itself.
      Example: `["projects/123456789", "folders/5432", "organizations/1234"]`
    assetType: The asset type of the resource as defined by CAIS. Example:
      `compute.googleapis.com/Firewall` See [Supported asset
      types](https://cloud.google.com/asset-inventory/docs/supported-asset-
      types) for more information.
    resource: The full name of the resource. Example: `//compute.googleapis.co
      m/projects/my_project_123/zones/zone1/instances/instance1` See [Resource
      names](https://cloud.google.com/apis/design/resource_names#full_resource
      _name) for more information.
  """
    ancestors = _messages.StringField(1, repeated=True)
    assetType = _messages.StringField(2)
    resource = _messages.StringField(3)