from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsDeleteRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsDeleteRequest object.

  Fields:
    name: Required. The name of the node group to delete. Format: projects/{pr
      oject}/regions/{region}/clusters/{cluster}/nodeGroups/{nodeGroup}
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two DeleteNodeGroupRequest (https://cloud.google.com/dat
      aproc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.
      v1.DeleteNodeGroupRequests) with the same ID, the second request is
      ignored and the first google.longrunning.Operation created and stored in
      the backend is returned.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)