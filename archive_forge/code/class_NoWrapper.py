from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NoWrapper(_messages.Message):
    """Sets the `data` field as the HTTP body for delivery.

  Fields:
    writeMetadata: Optional. When true, writes the Pub/Sub message metadata to
      `x-goog-pubsub-:` headers of the HTTP request. Writes the Pub/Sub
      message attributes to `:` headers of the HTTP request.
  """
    writeMetadata = _messages.BooleanField(1)