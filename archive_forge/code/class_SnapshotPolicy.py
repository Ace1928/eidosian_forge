from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotPolicy(_messages.Message):
    """Snapshot policies to take a snapshot of the existing dataflow job before
  beginning the deployment.

  Fields:
    description: User specified description of the snapshot. Maybe empty.
    snapshot: If true, task a snapshot of the existing dataflow job before
      beginning the deployment.
    snapshotSources: If true, perform snapshots for sources which support
      this.
    ttl: TTL for the snapshot.
  """
    description = _messages.StringField(1)
    snapshot = _messages.BooleanField(2)
    snapshotSources = _messages.BooleanField(3)
    ttl = _messages.StringField(4)