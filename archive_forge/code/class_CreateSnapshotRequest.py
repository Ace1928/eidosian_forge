from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CreateSnapshotRequest(_messages.Message):
    """Request for the `CreateSnapshot` method.

  Messages:
    LabelsValue: Optional. See [Creating and managing
      labels](https://cloud.google.com/pubsub/docs/labels).

  Fields:
    labels: Optional. See [Creating and managing
      labels](https://cloud.google.com/pubsub/docs/labels).
    subscription: Required. The subscription whose backlog the snapshot
      retains. Specifically, the created snapshot is guaranteed to retain: (a)
      The existing backlog on the subscription. More precisely, this is
      defined as the messages in the subscription's backlog that are
      unacknowledged upon the successful completion of the `CreateSnapshot`
      request; as well as: (b) Any messages published to the subscription's
      topic following the successful completion of the CreateSnapshot request.
      Format is `projects/{project}/subscriptions/{sub}`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. See [Creating and managing
    labels](https://cloud.google.com/pubsub/docs/labels).

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    subscription = _messages.StringField(2)