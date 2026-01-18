from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworksecurityV1CertificateProvider(_messages.Message):
    """Specification of certificate provider. Defines the mechanism to obtain
  the certificate and private key for peer to peer authentication.

  Fields:
    certificateProviderInstance: The certificate provider instance
      specification that will be passed to the data plane, which will be used
      to load necessary credential information.
    grpcEndpoint: gRPC specific configuration to access the gRPC server to
      obtain the cert and private key.
  """
    certificateProviderInstance = _messages.MessageField('CertificateProviderInstance', 1)
    grpcEndpoint = _messages.MessageField('GoogleCloudNetworksecurityV1GrpcEndpoint', 2)