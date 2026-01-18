from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsRollbackRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsRollbackRequest
  object.

  Fields:
    googleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest: A
      GoogleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest resource to
      be passed as the request body.
    name: Required. The name of the AwsNodePool resource to rollback.
      `AwsNodePool` names are formatted as
      `projects//locations//awsClusters//awsNodePools/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    googleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest = _messages.MessageField('GoogleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest', 1)
    name = _messages.StringField(2, required=True)