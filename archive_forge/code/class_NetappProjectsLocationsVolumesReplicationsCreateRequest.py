from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsCreateRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsCreateRequest object.

  Fields:
    parent: Required. The NetApp volume to create the replications of, in the
      format `projects/{project_id}/locations/{location}/volumes/{volume_id}`
    replication: A Replication resource to be passed as the request body.
    replicationId: Required. ID of the replication to create. This value must
      start with a lowercase letter followed by up to 62 lowercase letters,
      numbers, or hyphens, and cannot end with a hyphen.
  """
    parent = _messages.StringField(1, required=True)
    replication = _messages.MessageField('Replication', 2)
    replicationId = _messages.StringField(3)