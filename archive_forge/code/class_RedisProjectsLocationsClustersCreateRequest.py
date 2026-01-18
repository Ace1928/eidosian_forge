from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsClustersCreateRequest(_messages.Message):
    """A RedisProjectsLocationsClustersCreateRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterId: Required. The logical name of the Redis cluster in the customer
      project with the following restrictions: * Must contain only lowercase
      letters, numbers, and hyphens. * Must start with a letter. * Must be
      between 1-63 characters. * Must end with a number or a letter. * Must be
      unique within the customer project / location
    parent: Required. The resource name of the cluster location using the
      form: `projects/{project_id}/locations/{location_id}` where
      `location_id` refers to a GCP region.
    requestId: Idempotent request UUID.
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)