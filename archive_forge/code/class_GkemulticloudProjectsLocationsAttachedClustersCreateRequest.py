from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAttachedClustersCreateRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAttachedClustersCreateRequest object.

  Fields:
    attachedClusterId: Required. A client provided ID the resource. Must be
      unique within the parent resource. The provided ID will be part of the
      AttachedCluster resource name formatted as
      `projects//locations//attachedClusters/`. Valid characters are `/a-z-/`.
      Cannot be longer than 63 characters.
    googleCloudGkemulticloudV1AttachedCluster: A
      GoogleCloudGkemulticloudV1AttachedCluster resource to be passed as the
      request body.
    parent: Required. The parent location where this AttachedCluster resource
      will be created. Location names are formatted as `projects//locations/`.
      See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      create the cluster.
  """
    attachedClusterId = _messages.StringField(1)
    googleCloudGkemulticloudV1AttachedCluster = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedCluster', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)