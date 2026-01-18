from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1CertificateAttributes(_messages.Message):
    """Stores information about a certificate.

  Enums:
    ValidationStateValueValuesEnum: Output only. Validation state of this
      certificate.

  Fields:
    certificateTemplate: The X.509 extension for CertificateTemplate.
    fingerprint: The encoded certificate fingerprint.
    issuer: The name of the issuer of this certificate.
    serialNumber: Serial number of the certificate, Example: "123456789".
    subject: The subject name of this certificate.
    thumbprint: The certificate thumbprint.
    validationState: Output only. Validation state of this certificate.
    validityExpirationTime: Certificate not valid at or after this timestamp.
    validityStartTime: Certificate not valid before this timestamp.
  """

    class ValidationStateValueValuesEnum(_messages.Enum):
        """Output only. Validation state of this certificate.

    Values:
      CERTIFICATE_VALIDATION_STATE_UNSPECIFIED: Default value.
      VALIDATION_SUCCESSFUL: Certificate validation was successful.
      VALIDATION_FAILED: Certificate validation failed.
    """
        CERTIFICATE_VALIDATION_STATE_UNSPECIFIED = 0
        VALIDATION_SUCCESSFUL = 1
        VALIDATION_FAILED = 2
    certificateTemplate = _messages.MessageField('GoogleAppsCloudidentityDevicesV1CertificateTemplate', 1)
    fingerprint = _messages.StringField(2)
    issuer = _messages.StringField(3)
    serialNumber = _messages.StringField(4)
    subject = _messages.StringField(5)
    thumbprint = _messages.StringField(6)
    validationState = _messages.EnumField('ValidationStateValueValuesEnum', 7)
    validityExpirationTime = _messages.StringField(8)
    validityStartTime = _messages.StringField(9)