from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PingSource(_messages.Message):
    """A PingSource object.

  Fields:
    apiVersion: The API version for this call such as
      "sources.knative.dev/v1beta1".
    kind: The kind of resource, in this case "PingSource".
    metadata: Metadata associated with this PingSource.
    spec: Spec defines the desired state of the PingSource.
    status: Status represents the current state of the PingSource. This data
      may be out of date.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('PingSourceSpec', 4)
    status = _messages.MessageField('PingSourceStatus', 5)