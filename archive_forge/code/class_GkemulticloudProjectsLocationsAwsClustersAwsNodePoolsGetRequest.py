from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsGetRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsGetRequest
  object.

  Fields:
    name: Required. The name of the AwsNodePool resource to describe.
      `AwsNodePool` names are formatted as
      `projects//locations//awsClusters//awsNodePools/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    name = _messages.StringField(1, required=True)