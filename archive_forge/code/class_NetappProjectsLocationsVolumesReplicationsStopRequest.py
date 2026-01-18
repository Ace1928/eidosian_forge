from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsStopRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsStopRequest object.

  Fields:
    name: Required. The resource name of the replication, in the format of pro
      jects/{project_id}/locations/{location}/volumes/{volume_id}/replications
      /{replication_id}.
    stopReplicationRequest: A StopReplicationRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    stopReplicationRequest = _messages.MessageField('StopReplicationRequest', 2)