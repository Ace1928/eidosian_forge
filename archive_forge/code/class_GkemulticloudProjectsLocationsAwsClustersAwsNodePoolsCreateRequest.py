from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsCreateRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsCreateRequest
  object.

  Fields:
    awsNodePoolId: Required. A client provided ID the resource. Must be unique
      within the parent resource. The provided ID will be part of the
      AwsNodePool resource name formatted as
      `projects//locations//awsClusters//awsNodePools/`. Valid characters are
      `/a-z-/`. Cannot be longer than 63 characters.
    googleCloudGkemulticloudV1AwsNodePool: A
      GoogleCloudGkemulticloudV1AwsNodePool resource to be passed as the
      request body.
    parent: Required. The AwsCluster resource where this node pool will be
      created. `AwsCluster` names are formatted as
      `projects//locations//awsClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      create the node pool.
  """
    awsNodePoolId = _messages.StringField(1)
    googleCloudGkemulticloudV1AwsNodePool = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodePool', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)