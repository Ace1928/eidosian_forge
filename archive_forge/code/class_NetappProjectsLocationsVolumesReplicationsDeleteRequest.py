from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsDeleteRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsDeleteRequest object.

  Fields:
    name: Required. The replication resource name, in the format
      `projects/*/locations/*/volumes/*/replications/{replication_id}`
  """
    name = _messages.StringField(1, required=True)