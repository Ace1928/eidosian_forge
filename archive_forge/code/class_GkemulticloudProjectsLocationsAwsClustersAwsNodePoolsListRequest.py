from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsListRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsListRequest
  object.

  Fields:
    pageSize: The maximum number of items to return. If not specified, a
      default value of 50 will be used by the service. Regardless of the
      pageSize value, the response can include a partial list and a caller
      should only rely on response's nextPageToken to determine if there are
      more instances left to be queried.
    pageToken: The `nextPageToken` value returned from a previous
      awsNodePools.list request, if any.
    parent: Required. The parent `AwsCluster` which owns this collection of
      AwsNodePool resources. `AwsCluster` names are formatted as
      `projects//locations//awsClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)