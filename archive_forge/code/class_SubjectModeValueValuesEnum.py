from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubjectModeValueValuesEnum(_messages.Enum):
    """Immutable. Specifies how the Certificate's identity fields are to be
    decided. If this is omitted, the `DEFAULT` subject mode will be used.

    Values:
      SUBJECT_REQUEST_MODE_UNSPECIFIED: Not specified.
      DEFAULT: The default mode used in most cases. Indicates that the
        certificate's Subject and/or SubjectAltNames are specified in the
        certificate request. This mode requires the caller to have the
        `privateca.certificates.create` permission.
      REFLECTED_SPIFFE: A mode reserved for special cases. Indicates that the
        certificate should have one SPIFFE SubjectAltNames set by the service
        based on the caller's identity. This mode will ignore any explicitly
        specified Subject and/or SubjectAltNames in the certificate request.
        This mode requires the caller to have the
        `privateca.certificates.createForSelf` permission.
    """
    SUBJECT_REQUEST_MODE_UNSPECIFIED = 0
    DEFAULT = 1
    REFLECTED_SPIFFE = 2