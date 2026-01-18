from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudSchedulerSourceStatus(_messages.Message):
    """CloudSchedulerSourceStatus represents the current state of a
  CloudSchedulerSource.

  Messages:
    AnnotationsValue: Annotations is additional Status fields for the Resource
      to save some additional State as well as convey more information to the
      user. This is roughly akin to Annotations on any k8s resource, just the
      reconciler conveying richer information outwards.

  Fields:
    annotations: Annotations is additional Status fields for the Resource to
      save some additional State as well as convey more information to the
      user. This is roughly akin to Annotations on any k8s resource, just the
      reconciler conveying richer information outwards.
    ceAttributes: CloudEventAttributes are the specific attributes that the
      Source uses as part of its CloudEvents.
    conditions: Array of observed CloudSchedulerSourceConditions, indicating
      the current state of the CloudSchedulerSource.
    jobName: JobName is the name of the created scheduler Job on success.
    observedGeneration: ObservedGeneration is the 'Generation' of the
      CloudSchedulerSource that was last processed by the controller.
    projectId: ProjectID is the project ID of the Topic, might have been
      resolved.
    sinkUri: SinkURI is the current active sink URI that has been configured
      for the Source.
    subscriptionId: SubscriptionID is the created subscription ID.
    topicId: TopicID where the notifications are sent to.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations is additional Status fields for the Resource to save some
    additional State as well as convey more information to the user. This is
    roughly akin to Annotations on any k8s resource, just the reconciler
    conveying richer information outwards.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    ceAttributes = _messages.MessageField('CloudEventAttributes', 2, repeated=True)
    conditions = _messages.MessageField('Condition', 3, repeated=True)
    jobName = _messages.StringField(4)
    observedGeneration = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    projectId = _messages.StringField(6)
    sinkUri = _messages.StringField(7)
    subscriptionId = _messages.StringField(8)
    topicId = _messages.StringField(9)