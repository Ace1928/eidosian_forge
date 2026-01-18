from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApiServerSource(_messages.Message):
    """A ApiServerSource object.

  Fields:
    apiVersion: The API version for this call such as
      "sources.knative.dev/v1beta1".
    kind: The kind of resource, in this case "ApiServerSource".
    metadata: Metadata associated with this ApiServerSource.
    spec: Spec defines the desired state of the ApiServerSource.
    status: Status represents the current state of the ApiServerSource. This
      data may be out of date. +optional
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('ApiServerSourceSpec', 4)
    status = _messages.MessageField('ApiServerSourceStatus', 5)