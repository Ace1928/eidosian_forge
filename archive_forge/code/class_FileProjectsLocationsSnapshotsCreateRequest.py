from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsSnapshotsCreateRequest(_messages.Message):
    """A FileProjectsLocationsSnapshotsCreateRequest object.

  Fields:
    parent: Required. The snapshot's project and location, in the format
      `projects/{project_number}/locations/{location}`. In Filestore, snapshot
      locations map to GCP zones, for example **us-west1-b**, for local
      snapshots and to GCP regions, for example **us-west1**, otherwise.
    snapshot: A Snapshot resource to be passed as the request body.
    snapshotId: Required. The ID to use for the snapshot. The ID must be
      unique within the specified project and location. This value must start
      with a lowercase letter followed by up to 62 lowercase letters, numbers,
      or hyphens, and cannot end with a hyphen.
  """
    parent = _messages.StringField(1, required=True)
    snapshot = _messages.MessageField('Snapshot', 2)
    snapshotId = _messages.StringField(3)