from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotJobRequest(_messages.Message):
    """Request to create a snapshot of a job.

  Fields:
    description: User specified description of the snapshot. Maybe empty.
    location: The location that contains this job.
    snapshotSources: If true, perform snapshots for sources which support
      this.
    ttl: TTL for the snapshot.
  """
    description = _messages.StringField(1)
    location = _messages.StringField(2)
    snapshotSources = _messages.BooleanField(3)
    ttl = _messages.StringField(4)