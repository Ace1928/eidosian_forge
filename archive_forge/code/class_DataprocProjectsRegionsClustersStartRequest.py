from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersStartRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersStartRequest object.

  Fields:
    clusterName: Required. The cluster name.
    projectId: Required. The ID of the Google Cloud Platform project the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
    startClusterRequest: A StartClusterRequest resource to be passed as the
      request body.
  """
    clusterName = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    startClusterRequest = _messages.MessageField('StartClusterRequest', 4)