from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesPromoteReplicaRequest(_messages.Message):
    """A FileProjectsLocationsInstancesPromoteReplicaRequest object.

  Fields:
    name: Required. The resource name of the instance, in the format
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`.
    promoteReplicaRequest: A PromoteReplicaRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    promoteReplicaRequest = _messages.MessageField('PromoteReplicaRequest', 2)