from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnmanagedKafkaConfig(_messages.Message):
    """Config for customer provided Kafka to receive application logs from log
  forwarder. This field is only populated for LCP clusters.

  Fields:
    brokers: Required. Comma separated string of broker addresses, with IP and
      port.
    topicKey: Optional. Kafka topic key to select a topic if multiple topics
      exist.
    topics: Required. Comma separated string of Kafka topics.
  """
    brokers = _messages.StringField(1)
    topicKey = _messages.StringField(2)
    topics = _messages.StringField(3)