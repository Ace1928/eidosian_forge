from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsResumeRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsResumeRequest object.

  Fields:
    name: Required. The resource name of the replication, in the format of pro
      jects/{project_id}/locations/{location}/volumes/{volume_id}/replications
      /{replication_id}.
    resumeReplicationRequest: A ResumeReplicationRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    resumeReplicationRequest = _messages.MessageField('ResumeReplicationRequest', 2)