from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SeekRequest(_messages.Message):
    """Request for the `Seek` method.

  Fields:
    snapshot: Optional. The snapshot to seek to. The snapshot's topic must be
      the same as that of the provided subscription. Format is
      `projects/{project}/snapshots/{snap}`.
    time: Optional. The time to seek to. Messages retained in the subscription
      that were published before this time are marked as acknowledged, and
      messages retained in the subscription that were published after this
      time are marked as unacknowledged. Note that this operation affects only
      those messages retained in the subscription (configured by the
      combination of `message_retention_duration` and
      `retain_acked_messages`). For example, if `time` corresponds to a point
      before the message retention window (or to a point before the system's
      notion of the subscription creation time), only retained messages will
      be marked as unacknowledged, and already-expunged messages will not be
      restored.
  """
    snapshot = _messages.StringField(1)
    time = _messages.StringField(2)