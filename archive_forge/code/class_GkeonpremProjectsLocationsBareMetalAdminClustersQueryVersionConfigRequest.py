from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest(_messages.Message):
    """A
  GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest
  object.

  Fields:
    parent: Required. The parent of the project and location to query for
      version config. Format: "projects/{project}/locations/{location}"
    upgradeConfig_clusterName: The admin cluster resource name. This is the
      full resource name of the admin cluster resource. Format: "projects/{pro
      ject}/locations/{location}/bareMetalAdminClusters/{bare_metal_admin_clus
      ter}"
  """
    parent = _messages.StringField(1, required=True)
    upgradeConfig_clusterName = _messages.StringField(2)