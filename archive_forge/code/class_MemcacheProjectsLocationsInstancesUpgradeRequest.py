from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesUpgradeRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesUpgradeRequest object.

  Fields:
    googleCloudMemcacheV1UpgradeInstanceRequest: A
      GoogleCloudMemcacheV1UpgradeInstanceRequest resource to be passed as the
      request body.
    name: Required. Memcache instance resource name using the form:
      `projects/{project}/locations/{location}/instances/{instance}` where
      `location_id` refers to a GCP region.
  """
    googleCloudMemcacheV1UpgradeInstanceRequest = _messages.MessageField('GoogleCloudMemcacheV1UpgradeInstanceRequest', 1)
    name = _messages.StringField(2, required=True)