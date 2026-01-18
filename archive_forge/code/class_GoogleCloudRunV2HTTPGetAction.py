from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2HTTPGetAction(_messages.Message):
    """HTTPGetAction describes an action based on HTTP Get requests.

  Fields:
    httpHeaders: Optional. Custom headers to set in the request. HTTP allows
      repeated headers.
    path: Optional. Path to access on the HTTP server. Defaults to '/'.
    port: Optional. Port number to access on the container. Must be in the
      range 1 to 65535. If not specified, defaults to the exposed port of the
      container, which is the value of container.ports[0].containerPort.
  """
    httpHeaders = _messages.MessageField('GoogleCloudRunV2HTTPHeader', 1, repeated=True)
    path = _messages.StringField(2)
    port = _messages.IntegerField(3, variant=_messages.Variant.INT32)