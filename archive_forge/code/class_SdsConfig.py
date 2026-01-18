from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SdsConfig(_messages.Message):
    """[Deprecated] The configuration to access the SDS server. The
  configuration to access the SDS server.

  Fields:
    grpcServiceConfig: The configuration to access the SDS server over GRPC.
  """
    grpcServiceConfig = _messages.MessageField('GrpcServiceConfig', 1)