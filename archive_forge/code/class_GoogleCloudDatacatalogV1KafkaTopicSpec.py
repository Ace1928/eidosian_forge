from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1KafkaTopicSpec(_messages.Message):
    """Entry specification of a Kafka topic.

  Fields:
    clusterEntry: Required. Name of the Kafka cluster entry this topic is a
      part of. Example:
      `projects/my_project/locations/us/entryGroups/kafka/entries/my_cluster`.
      Data Catalog doesn't validate the content of this field.
    topic: Required. Name of the Kafka topic this entry represents. Example:
      `my_topic`. Data Catalog doesn't validate the content of this field.
  """
    clusterEntry = _messages.StringField(1)
    topic = _messages.StringField(2)