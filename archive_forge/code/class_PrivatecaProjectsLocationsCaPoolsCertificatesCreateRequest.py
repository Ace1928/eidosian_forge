from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificatesCreateRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificatesCreateRequest object.

  Fields:
    certificate: A Certificate resource to be passed as the request body.
    certificateId: Optional. It must be unique within a location and match the
      regular expression `[a-zA-Z0-9_-]{1,63}`. This field is required when
      using a CertificateAuthority in the Enterprise
      CertificateAuthority.Tier, but is optional and its value is ignored
      otherwise.
    issuingCertificateAuthorityId: Optional. The resource ID of the
      CertificateAuthority that should issue the certificate. This optional
      field will ignore the load-balancing scheme of the Pool and directly
      issue the certificate from the CA with the specified ID, contained in
      the same CaPool referenced by `parent`. Per-CA quota rules apply. If
      left empty, a CertificateAuthority will be chosen from the CaPool by the
      service. For example, to issue a Certificate from a Certificate
      Authority with resource name "projects/my-project/locations/us-
      central1/caPools/my-pool/certificateAuthorities/my-ca", you can set the
      parent to "projects/my-project/locations/us-central1/caPools/my-pool"
      and the issuing_certificate_authority_id to "my-ca".
    parent: Required. The resource name of the CaPool associated with the
      Certificate, in the format `projects/*/locations/*/caPools/*`.
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
    validateOnly: Optional. If this is true, no Certificate resource will be
      persisted regardless of the CaPool's tier, and the returned Certificate
      will not contain the pem_certificate field.
  """
    certificate = _messages.MessageField('Certificate', 1)
    certificateId = _messages.StringField(2)
    issuingCertificateAuthorityId = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)