from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDeleteRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDeleteRequest
  object.

  Fields:
    ignoreActiveCertificates: Optional. This field allows the CA to be deleted
      even if the CA has active certs. Active certs include both unrevoked and
      unexpired certs.
    ignoreDependentResources: Optional. This field allows this CA to be
      deleted even if it's being depended on by another resource. However,
      doing so may result in unintended and unrecoverable effects on any
      dependent resources since the CA will no longer be able to issue
      certificates.
    name: Required. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
    requestId: Optional. An ID to identify requests. Specify a unique request
      ID so that if you must retry your request, the server will know to
      ignore the request if it has already been completed. The server will
      guarantee that for at least 60 minutes since the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    skipGracePeriod: Optional. If this flag is set, the Certificate Authority
      will be deleted as soon as possible without a 30-day grace period where
      undeletion would have been allowed. If you proceed, there will be no way
      to recover this CA.
  """
    ignoreActiveCertificates = _messages.BooleanField(1)
    ignoreDependentResources = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    skipGracePeriod = _messages.BooleanField(5)