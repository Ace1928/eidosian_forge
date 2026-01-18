from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeCluster(_messages.Message):
    """GkeCluster contains information specific to GKE clusters.

  Fields:
    clusterMissing: Output only. If cluster_missing is set then it denotes
      that the GKE cluster no longer exists in the GKE Control Plane.
    resourceLink: Immutable. Self-link of the Google Cloud resource for the
      GKE cluster. For example: //container.googleapis.com/projects/my-
      project/locations/us-west1-a/clusters/my-cluster Zonal clusters are also
      supported.
  """
    clusterMissing = _messages.BooleanField(1)
    resourceLink = _messages.StringField(2)