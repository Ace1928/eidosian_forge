from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateExtensionConstraints(_messages.Message):
    """Describes a set of X.509 extensions that may be part of some certificate
  issuance controls.

  Enums:
    KnownExtensionsValueListEntryValuesEnum:

  Fields:
    additionalExtensions: Optional. A set of ObjectIds identifying custom
      X.509 extensions. Will be combined with known_extensions to determine
      the full set of X.509 extensions.
    knownExtensions: Optional. A set of named X.509 extensions. Will be
      combined with additional_extensions to determine the full set of X.509
      extensions.
  """

    class KnownExtensionsValueListEntryValuesEnum(_messages.Enum):
        """KnownExtensionsValueListEntryValuesEnum enum type.

    Values:
      KNOWN_CERTIFICATE_EXTENSION_UNSPECIFIED: Not specified.
      BASE_KEY_USAGE: Refers to a certificate's Key Usage extension, as
        described in [RFC 5280 section
        4.2.1.3](https://tools.ietf.org/html/rfc5280#section-4.2.1.3). This
        corresponds to the KeyUsage.base_key_usage field.
      EXTENDED_KEY_USAGE: Refers to a certificate's Extended Key Usage
        extension, as described in [RFC 5280 section
        4.2.1.12](https://tools.ietf.org/html/rfc5280#section-4.2.1.12). This
        corresponds to the KeyUsage.extended_key_usage message.
      CA_OPTIONS: Refers to a certificate's Basic Constraints extension, as
        described in [RFC 5280 section
        4.2.1.9](https://tools.ietf.org/html/rfc5280#section-4.2.1.9). This
        corresponds to the X509Parameters.ca_options field.
      POLICY_IDS: Refers to a certificate's Policy object identifiers, as
        described in [RFC 5280 section
        4.2.1.4](https://tools.ietf.org/html/rfc5280#section-4.2.1.4). This
        corresponds to the X509Parameters.policy_ids field.
      AIA_OCSP_SERVERS: Refers to OCSP servers in a certificate's Authority
        Information Access extension, as described in [RFC 5280 section
        4.2.2.1](https://tools.ietf.org/html/rfc5280#section-4.2.2.1), This
        corresponds to the X509Parameters.aia_ocsp_servers field.
      NAME_CONSTRAINTS: Refers to Name Constraints extension as described in
        [RFC 5280 section
        4.2.1.10](https://tools.ietf.org/html/rfc5280#section-4.2.1.10)
    """
        KNOWN_CERTIFICATE_EXTENSION_UNSPECIFIED = 0
        BASE_KEY_USAGE = 1
        EXTENDED_KEY_USAGE = 2
        CA_OPTIONS = 3
        POLICY_IDS = 4
        AIA_OCSP_SERVERS = 5
        NAME_CONSTRAINTS = 6
    additionalExtensions = _messages.MessageField('ObjectId', 1, repeated=True)
    knownExtensions = _messages.EnumField('KnownExtensionsValueListEntryValuesEnum', 2, repeated=True)