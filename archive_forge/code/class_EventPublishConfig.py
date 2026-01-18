from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventPublishConfig(_messages.Message):
    """Confirguration of PubSubEventWriter.

  Fields:
    enabled: Required. Option to enable Event Publishing.
    topic: Required. The resource name of the Pub/Sub topic. Format:
      projects/{project_id}/topics/{topic_id}
  """
    enabled = _messages.BooleanField(1)
    topic = _messages.StringField(2)