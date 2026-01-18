from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsGetRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsGetRequest object.

  Fields:
    name: Required. The name of the node group to retrieve. Format: projects/{
      project}/regions/{region}/clusters/{cluster}/nodeGroups/{nodeGroup}
  """
    name = _messages.StringField(1, required=True)