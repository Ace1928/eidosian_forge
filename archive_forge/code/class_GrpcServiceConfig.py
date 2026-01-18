from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcServiceConfig(_messages.Message):
    """[Deprecated] gRPC config to access the SDS server. gRPC config to access
  the SDS server.

  Fields:
    callCredentials: The call credentials to access the SDS server.
    channelCredentials: The channel credentials to access the SDS server.
    targetUri: The target URI of the SDS server.
  """
    callCredentials = _messages.MessageField('CallCredentials', 1)
    channelCredentials = _messages.MessageField('ChannelCredentials', 2)
    targetUri = _messages.StringField(3)