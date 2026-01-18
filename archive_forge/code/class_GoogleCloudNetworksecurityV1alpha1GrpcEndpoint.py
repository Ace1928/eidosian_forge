from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworksecurityV1alpha1GrpcEndpoint(_messages.Message):
    """Specification of the GRPC Endpoint.

  Fields:
    targetUri: Required. The target URI of the gRPC endpoint. Only UDS path is
      supported, and should start with "unix:".
  """
    targetUri = _messages.StringField(1)