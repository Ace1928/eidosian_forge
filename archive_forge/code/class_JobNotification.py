from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobNotification(_messages.Message):
    """Notification configurations.

  Fields:
    message: The attribute requirements of messages to be sent to this Pub/Sub
      topic. Without this field, no message will be sent.
    pubsubTopic: The Pub/Sub topic where notifications like the job state
      changes will be published. The topic must exist in the same project as
      the job and billings will be charged to this project. If not specified,
      no Pub/Sub messages will be sent. Topic format:
      `projects/{project}/topics/{topic}`.
  """
    message = _messages.MessageField('Message', 1)
    pubsubTopic = _messages.StringField(2)