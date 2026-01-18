from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DsRecord(_messages.Message):
    """Defines a Delegation Signer (DS) record, which is needed to enable
  DNSSEC for a domain. It contains a digest (hash) of a DNSKEY record that
  must be present in the domain's DNS zone.

  Enums:
    AlgorithmValueValuesEnum: The algorithm used to generate the referenced
      DNSKEY.
    DigestTypeValueValuesEnum: The hash function used to generate the digest
      of the referenced DNSKEY.

  Fields:
    algorithm: The algorithm used to generate the referenced DNSKEY.
    digest: The digest generated from the referenced DNSKEY.
    digestType: The hash function used to generate the digest of the
      referenced DNSKEY.
    keyTag: The key tag of the record. Must be set in range 0 -- 65535.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """The algorithm used to generate the referenced DNSKEY.

    Values:
      ALGORITHM_UNSPECIFIED: The algorithm is unspecified.
      RSAMD5: RSA/MD5. Cannot be used for new deployments.
      DH: Diffie-Hellman. Cannot be used for new deployments.
      DSA: DSA/SHA1. Not recommended for new deployments.
      ECC: ECC. Not recommended for new deployments.
      RSASHA1: RSA/SHA-1. Not recommended for new deployments.
      DSANSEC3SHA1: DSA-NSEC3-SHA1. Not recommended for new deployments.
      RSASHA1NSEC3SHA1: RSA/SHA1-NSEC3-SHA1. Not recommended for new
        deployments.
      RSASHA256: RSA/SHA-256.
      RSASHA512: RSA/SHA-512.
      ECCGOST: GOST R 34.10-2001.
      ECDSAP256SHA256: ECDSA Curve P-256 with SHA-256.
      ECDSAP384SHA384: ECDSA Curve P-384 with SHA-384.
      ED25519: Ed25519.
      ED448: Ed448.
      INDIRECT: Reserved for Indirect Keys. Cannot be used for new
        deployments.
      PRIVATEDNS: Private algorithm. Cannot be used for new deployments.
      PRIVATEOID: Private algorithm OID. Cannot be used for new deployments.
    """
        ALGORITHM_UNSPECIFIED = 0
        RSAMD5 = 1
        DH = 2
        DSA = 3
        ECC = 4
        RSASHA1 = 5
        DSANSEC3SHA1 = 6
        RSASHA1NSEC3SHA1 = 7
        RSASHA256 = 8
        RSASHA512 = 9
        ECCGOST = 10
        ECDSAP256SHA256 = 11
        ECDSAP384SHA384 = 12
        ED25519 = 13
        ED448 = 14
        INDIRECT = 15
        PRIVATEDNS = 16
        PRIVATEOID = 17

    class DigestTypeValueValuesEnum(_messages.Enum):
        """The hash function used to generate the digest of the referenced
    DNSKEY.

    Values:
      DIGEST_TYPE_UNSPECIFIED: The DigestType is unspecified.
      SHA1: SHA-1. Not recommended for new deployments.
      SHA256: SHA-256.
      GOST3411: GOST R 34.11-94.
      SHA384: SHA-384.
    """
        DIGEST_TYPE_UNSPECIFIED = 0
        SHA1 = 1
        SHA256 = 2
        GOST3411 = 3
        SHA384 = 4
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    digest = _messages.StringField(2)
    digestType = _messages.EnumField('DigestTypeValueValuesEnum', 3)
    keyTag = _messages.IntegerField(4, variant=_messages.Variant.INT32)