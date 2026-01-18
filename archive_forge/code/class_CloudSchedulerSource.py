from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudSchedulerSource(_messages.Message):
    """The CloudSchedulerSource resource.

  Fields:
    apiVersion: The API version for this call such as
      "events.cloud.google.com/v1".
    kind: The kind of resource, in this case "CloudSchedulerSource".
    metadata: Metadata associated with this CloudSchedulerSource.
    spec: Spec defines the desired state of the CloudSchedulerSource.
    status: Status represents the current state of the CloudSchedulerSource.
      This data may be out of date.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('CloudSchedulerSourceSpec', 4)
    status = _messages.MessageField('CloudSchedulerSourceStatus', 5)