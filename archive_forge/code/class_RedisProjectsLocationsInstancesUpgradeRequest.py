from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsInstancesUpgradeRequest(_messages.Message):
    """A RedisProjectsLocationsInstancesUpgradeRequest object.

  Fields:
    name: Required. Redis instance resource name using the form:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      where `location_id` refers to a GCP region.
    upgradeInstanceRequest: A UpgradeInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    upgradeInstanceRequest = _messages.MessageField('UpgradeInstanceRequest', 2)