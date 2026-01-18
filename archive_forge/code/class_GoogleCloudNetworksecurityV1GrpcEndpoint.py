from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworksecurityV1GrpcEndpoint(_messages.Message):
    """Specification of the GRPC Endpoint.

  Fields:
    sdsResource: Optional. sds_resource is used to set the name of the SDS
      configuration. When used in the context of GSM, the following rules
      apply If the resource name is "default" and "ROOTCA" then it implies
      ISTIO_MUTUAL tlsMode. If the resource name begins with "file-cert"
      and/or "file-root", it implies custom MUTUAL tlsMode
    targetUri: Required. The target URI of the gRPC endpoint. Only UDS path is
      supported, and should start with "unix:".
  """
    sdsResource = _messages.StringField(1)
    targetUri = _messages.StringField(2)