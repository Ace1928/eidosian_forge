from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubLocation(_messages.Message):
    """Identifies a pubsub location to use for transferring data into or out of
  a streaming Dataflow job.

  Fields:
    dropLateData: Indicates whether the pipeline allows late-arriving data.
    dynamicDestinations: If true, then this location represents dynamic
      topics.
    idLabel: If set, contains a pubsub label from which to extract record ids.
      If left empty, record deduplication will be strictly best effort.
    subscription: A pubsub subscription, in the form of
      "pubsub.googleapis.com/subscriptions//"
    timestampLabel: If set, contains a pubsub label from which to extract
      record timestamps. If left empty, record timestamps will be generated
      upon arrival.
    topic: A pubsub topic, in the form of "pubsub.googleapis.com/topics//"
    trackingSubscription: If set, specifies the pubsub subscription that will
      be used for tracking custom time timestamps for watermark estimation.
    withAttributes: If true, then the client has requested to get pubsub
      attributes.
  """
    dropLateData = _messages.BooleanField(1)
    dynamicDestinations = _messages.BooleanField(2)
    idLabel = _messages.StringField(3)
    subscription = _messages.StringField(4)
    timestampLabel = _messages.StringField(5)
    topic = _messages.StringField(6)
    trackingSubscription = _messages.StringField(7)
    withAttributes = _messages.BooleanField(8)