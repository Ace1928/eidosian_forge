from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersGetJwksRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersGetJwksRequest object.

  Fields:
    awsCluster: Required. The AwsCluster, which owns the JsonWebKeys. Format:
      projects/{project}/locations/{location}/awsClusters/{cluster}
  """
    awsCluster = _messages.StringField(1, required=True)