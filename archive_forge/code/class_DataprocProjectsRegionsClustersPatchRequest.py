from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersPatchRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersPatchRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterName: Required. The cluster name.
    gracefulDecommissionTimeout: Optional. Timeout for graceful YARN
      decommissioning. Graceful decommissioning allows removing nodes from the
      cluster without interrupting jobs in progress. Timeout specifies how
      long to wait for jobs in progress to finish before forcefully removing
      nodes (and potentially interrupting jobs). Default timeout is 0 (for
      forceful decommission), and the maximum allowed timeout is 1 day. (see
      JSON representation of Duration (https://developers.google.com/protocol-
      buffers/docs/proto3#json)).Only supported on Dataproc image versions 1.2
      and higher.
    projectId: Required. The ID of the Google Cloud Platform project the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two UpdateClusterRequest (https://cloud.google.com/datap
      roc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1
      .UpdateClusterRequest)s with the same id, then the second request will
      be ignored and the first google.longrunning.Operation created and stored
      in the backend is returned.It is recommended to always set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    updateMask: Required. Specifies the path, relative to Cluster, of the
      field to update. For example, to change the number of workers in a
      cluster to 5, the update_mask parameter would be specified as
      config.worker_config.num_instances, and the PATCH request body would
      specify the new value, as follows: { "config":{ "workerConfig":{
      "numInstances":"5" } } } Similarly, to change the number of preemptible
      workers in a cluster to 5, the update_mask parameter would be
      config.secondary_worker_config.num_instances, and the PATCH request body
      would be set as follows: { "config":{ "secondaryWorkerConfig":{
      "numInstances":"5" } } } *Note:* Currently, only the following fields
      can be updated: *Mask* *Purpose* *labels* Update labels
      *config.worker_config.num_instances* Resize primary worker group
      *config.secondary_worker_config.num_instances* Resize secondary worker
      group config.autoscaling_config.policy_uri Use, stop using, or change
      autoscaling policies
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterName = _messages.StringField(2, required=True)
    gracefulDecommissionTimeout = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    region = _messages.StringField(5, required=True)
    requestId = _messages.StringField(6)
    updateMask = _messages.StringField(7)