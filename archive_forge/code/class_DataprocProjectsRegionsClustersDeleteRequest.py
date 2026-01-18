from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersDeleteRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersDeleteRequest object.

  Fields:
    clusterName: Required. The cluster name.
    clusterUuid: Optional. Specifying the cluster_uuid means the RPC should
      fail (with error NOT_FOUND) if cluster with specified UUID does not
      exist.
    gracefulTerminationTimeout: Optional. The graceful termination timeout for
      the deletion of the cluster. Indicate the time the request will wait to
      complete the running jobs on the cluster before its forceful deletion.
      Default value is 0 indicating that the user has not enabled the graceful
      termination. Value can be between 60 second and 6 Hours, in case the
      graceful termination is enabled. (There is no separate flag to check the
      enabling or disabling of graceful termination, it can be checked by the
      values in the field).
    projectId: Required. The ID of the Google Cloud Platform project that the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two DeleteClusterRequest (https://cloud.google.com/datap
      roc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1
      .DeleteClusterRequest)s with the same id, then the second request will
      be ignored and the first google.longrunning.Operation created and stored
      in the backend is returned.It is recommended to always set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    clusterName = _messages.StringField(1, required=True)
    clusterUuid = _messages.StringField(2)
    gracefulTerminationTimeout = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    region = _messages.StringField(5, required=True)
    requestId = _messages.StringField(6)