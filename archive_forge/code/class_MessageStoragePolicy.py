from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MessageStoragePolicy(_messages.Message):
    """A policy constraining the storage of messages published to the topic.

  Fields:
    allowedPersistenceRegions: Optional. A list of IDs of Google Cloud regions
      where messages that are published to the topic may be persisted in
      storage. Messages published by publishers running in non-allowed Google
      Cloud regions (or running outside of Google Cloud altogether) are routed
      for storage in one of the allowed regions. An empty list means that no
      regions are allowed, and is not a valid configuration.
    enforceInTransit: Optional. If true, `allowed_persistence_regions` is also
      used to enforce in-transit guarantees for messages. That is, Pub/Sub
      will fail Publish operations on this topic and subscribe operations on
      any subscription attached to this topic in any region that is not in
      `allowed_persistence_regions`.
  """
    allowedPersistenceRegions = _messages.StringField(1, repeated=True)
    enforceInTransit = _messages.BooleanField(2)