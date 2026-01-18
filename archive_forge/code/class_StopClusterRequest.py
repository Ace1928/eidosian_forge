from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopClusterRequest(_messages.Message):
    """A request to stop a cluster.

  Fields:
    clusterUuid: Optional. Specifying the cluster_uuid means the RPC will fail
      (with error NOT_FOUND) if a cluster with the specified UUID does not
      exist.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two StopClusterRequest (https://cloud.google.com/datapro
      c/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1.S
      topClusterRequest)s with the same id, then the second request will be
      ignored and the first google.longrunning.Operation created and stored in
      the backend is returned.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    clusterUuid = _messages.StringField(1)
    requestId = _messages.StringField(2)