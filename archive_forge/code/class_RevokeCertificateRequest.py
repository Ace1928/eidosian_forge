from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevokeCertificateRequest(_messages.Message):
    """Request message for CertificateAuthorityService.RevokeCertificate.

  Enums:
    ReasonValueValuesEnum: Required. The RevocationReason for revoking this
      certificate.

  Fields:
    reason: Required. The RevocationReason for revoking this certificate.
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
  """

    class ReasonValueValuesEnum(_messages.Enum):
        """Required. The RevocationReason for revoking this certificate.

    Values:
      REVOCATION_REASON_UNSPECIFIED: Default unspecified value. This value
        does indicate that a Certificate has been revoked, but that a reason
        has not been recorded.
      KEY_COMPROMISE: Key material for this Certificate may have leaked.
      CERTIFICATE_AUTHORITY_COMPROMISE: The key material for a certificate
        authority in the issuing path may have leaked.
      AFFILIATION_CHANGED: The subject or other attributes in this Certificate
        have changed.
      SUPERSEDED: This Certificate has been superseded.
      CESSATION_OF_OPERATION: This Certificate or entities in the issuing path
        have ceased to operate.
      CERTIFICATE_HOLD: This Certificate should not be considered valid, it is
        expected that it may become valid in the future.
      PRIVILEGE_WITHDRAWN: This Certificate no longer has permission to assert
        the listed attributes.
      ATTRIBUTE_AUTHORITY_COMPROMISE: The authority which determines
        appropriate attributes for a Certificate may have been compromised.
    """
        REVOCATION_REASON_UNSPECIFIED = 0
        KEY_COMPROMISE = 1
        CERTIFICATE_AUTHORITY_COMPROMISE = 2
        AFFILIATION_CHANGED = 3
        SUPERSEDED = 4
        CESSATION_OF_OPERATION = 5
        CERTIFICATE_HOLD = 6
        PRIVILEGE_WITHDRAWN = 7
        ATTRIBUTE_AUTHORITY_COMPROMISE = 8
    reason = _messages.EnumField('ReasonValueValuesEnum', 1)
    requestId = _messages.StringField(2)