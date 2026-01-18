from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaProxyConfig(_messages.Message):
    """Proxy Configuration of a PartnerTenant.

  Fields:
    createTime: Output only. Timestamp when the resource was created.
    displayName: Optional. An arbitrary caller-provided name for the
      ProxyConfig. Cannot exceed 64 characters.
    encryptionInfo: Optional. Information to encrypt JWT for the proxy server.
    name: Output only. ProxyConfig resource name.
    proxyUri: Required. The URI of the proxy server.
    routingInfo: Required. Routing info to direct traffic to the proxy server.
    transportInfo: Optional. Transport layer information to verify for the
      proxy server.
    updateTime: Output only. Timestamp when the resource was last modified.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    encryptionInfo = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaEncryptionInfo', 3)
    name = _messages.StringField(4)
    proxyUri = _messages.StringField(5)
    routingInfo = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaRoutingInfo', 6)
    transportInfo = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaTransportInfo', 7)
    updateTime = _messages.StringField(8)