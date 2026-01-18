from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsClustersPatchRequest(_messages.Message):
    """A RedisProjectsLocationsClustersPatchRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    name: Required. Unique name of the resource in this scope including
      project and location using the form:
      `projects/{project_id}/locations/{location_id}/clusters/{cluster_id}`
    requestId: Idempotent request UUID.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields from Cluster: * `size_gb` * `replica_count`
  """
    cluster = _messages.MessageField('Cluster', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)