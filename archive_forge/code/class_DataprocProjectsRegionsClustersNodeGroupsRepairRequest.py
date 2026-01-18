from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsRepairRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsRepairRequest object.

  Fields:
    name: Required. The name of the node group to resize. Format: projects/{pr
      oject}/regions/{region}/clusters/{cluster}/nodeGroups/{nodeGroup}
    repairNodeGroupRequest: A RepairNodeGroupRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    repairNodeGroupRequest = _messages.MessageField('RepairNodeGroupRequest', 2)