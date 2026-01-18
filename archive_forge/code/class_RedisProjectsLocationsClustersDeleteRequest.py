from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsClustersDeleteRequest(_messages.Message):
    """A RedisProjectsLocationsClustersDeleteRequest object.

  Fields:
    name: Required. Redis cluster resource name using the form:
      `projects/{project_id}/locations/{location_id}/clusters/{cluster_id}`
      where `location_id` refers to a GCP region.
    requestId: Idempotent request UUID.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)