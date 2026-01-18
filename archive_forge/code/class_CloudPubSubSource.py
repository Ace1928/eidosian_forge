from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudPubSubSource(_messages.Message):
    """A CloudPubSubSource object.

  Fields:
    apiVersion: The API version for this call such as
      "events.cloud.google.com/v1".
    kind: The kind of resource, in this case "CloudPubSubSource".
    metadata: Metadata associated with this CloudPubSubSource.
    spec: Spec defines the desired state of the CloudPubSubSource.
    status: Status represents the current state of the CloudPubSubSource. This
      data may be out of date. +optional
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('CloudPubSubSourceSpec', 4)
    status = _messages.MessageField('CloudPubSubSourceStatus', 5)