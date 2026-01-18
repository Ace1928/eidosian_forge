from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersCreateRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersCreateRequest object.

  Enums:
    ActionOnFailedPrimaryWorkersValueValuesEnum: Optional. Failure action when
      primary worker creation fails.

  Fields:
    actionOnFailedPrimaryWorkers: Optional. Failure action when primary worker
      creation fails.
    cluster: A Cluster resource to be passed as the request body.
    projectId: Required. The ID of the Google Cloud Platform project that the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two CreateClusterRequest (https://cloud.google.com/datap
      roc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1
      .CreateClusterRequest)s with the same id, then the second request will
      be ignored and the first google.longrunning.Operation created and stored
      in the backend is returned.It is recommended to always set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """

    class ActionOnFailedPrimaryWorkersValueValuesEnum(_messages.Enum):
        """Optional. Failure action when primary worker creation fails.

    Values:
      FAILURE_ACTION_UNSPECIFIED: When FailureAction is unspecified, failure
        action defaults to NO_ACTION.
      NO_ACTION: Take no action on failure to create a cluster resource.
        NO_ACTION is the default.
      DELETE: Delete the failed cluster resource.
    """
        FAILURE_ACTION_UNSPECIFIED = 0
        NO_ACTION = 1
        DELETE = 2
    actionOnFailedPrimaryWorkers = _messages.EnumField('ActionOnFailedPrimaryWorkersValueValuesEnum', 1)
    cluster = _messages.MessageField('Cluster', 2)
    projectId = _messages.StringField(3, required=True)
    region = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)