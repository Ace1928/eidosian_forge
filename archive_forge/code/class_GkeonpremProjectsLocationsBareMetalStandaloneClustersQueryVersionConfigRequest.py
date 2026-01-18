from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfigRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfi
  gRequest object.

  Fields:
    parent: Required. The parent of the project and location to query for
      version config. Format: "projects/{project}/locations/{location}"
    upgradeConfig_clusterName: The standalone cluster resource name. This is
      the full resource name of the standalone cluster resource. Format: "proj
      ects/{project}/locations/{location}/bareMetalStandaloneClusters/{bare_me
      tal_standalone_cluster}"
  """
    parent = _messages.StringField(1, required=True)
    upgradeConfig_clusterName = _messages.StringField(2)