from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubTarget(_messages.Message):
    """Pub/Sub target. The job will be delivered by publishing a message to the
  given Pub/Sub topic.

  Messages:
    AttributesValue: Attributes for PubsubMessage.  Pubsub message must
      contain either non-empty data, or at least one attribute.

  Fields:
    attributes: Attributes for PubsubMessage.  Pubsub message must contain
      either non-empty data, or at least one attribute.
    data: The message payload for PubsubMessage.  Pubsub message must contain
      either non-empty data, or at least one attribute.
    topicName: Required.  The name of the Cloud Pub/Sub topic to which
      messages will be published when a job is delivered. The topic name must
      be in the same format as required by PubSub's [PublishRequest.name](http
      s://cloud.google.com/pubsub/docs/reference/rpc/google.pubsub.v1#publishr
      equest), for example `projects/PROJECT_ID/topics/TOPIC_ID`.  The topic
      must be in the same project as the Cloud Scheduler job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Attributes for PubsubMessage.  Pubsub message must contain either non-
    empty data, or at least one attribute.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributes = _messages.MessageField('AttributesValue', 1)
    data = _messages.BytesField(2)
    topicName = _messages.StringField(3)