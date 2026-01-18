from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsKey(_messages.Message):
    """A DNSSEC key pair.

  Enums:
    AlgorithmValueValuesEnum: String mnemonic specifying the DNSSEC algorithm
      of this key. Immutable after creation time.
    TypeValueValuesEnum: One of "KEY_SIGNING" or "ZONE_SIGNING". Keys of type
      KEY_SIGNING have the Secure Entry Point flag set and, when active, are
      used to sign only resource record sets of type DNSKEY. Otherwise, the
      Secure Entry Point flag is cleared, and this key is used to sign only
      resource record sets of other types. Immutable after creation time.

  Fields:
    algorithm: String mnemonic specifying the DNSSEC algorithm of this key.
      Immutable after creation time.
    creationTime: The time that this resource was created in the control
      plane. This is in RFC3339 text format. Output only.
    description: A mutable string of at most 1024 characters associated with
      this resource for the user's convenience. Has no effect on the
      resource's function.
    digests: Cryptographic hashes of the DNSKEY resource record associated
      with this DnsKey. These digests are needed to construct a DS record that
      points at this DNS key. Output only.
    id: Unique identifier for the resource; defined by the server (output
      only).
    isActive: Active keys are used to sign subsequent changes to the
      ManagedZone. Inactive keys are still present as DNSKEY Resource Records
      for the use of resolvers validating existing signatures.
    keyLength: Length of the key in bits. Specified at creation time, and then
      immutable.
    keyTag: The key tag is a non-cryptographic hash of the a DNSKEY resource
      record associated with this DnsKey. The key tag can be used to identify
      a DNSKEY more quickly (but it is not a unique identifier). In
      particular, the key tag is used in a parent zone's DS record to point at
      the DNSKEY in this child ManagedZone. The key tag is a number in the
      range [0, 65535] and the algorithm to calculate it is specified in
      RFC4034 Appendix B. Output only.
    kind: A string attribute.
    publicKey: Base64 encoded public half of this key. Output only.
    type: One of "KEY_SIGNING" or "ZONE_SIGNING". Keys of type KEY_SIGNING
      have the Secure Entry Point flag set and, when active, are used to sign
      only resource record sets of type DNSKEY. Otherwise, the Secure Entry
      Point flag is cleared, and this key is used to sign only resource record
      sets of other types. Immutable after creation time.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """String mnemonic specifying the DNSSEC algorithm of this key. Immutable
    after creation time.

    Values:
      rsasha1: <no description>
      rsasha256: <no description>
      rsasha512: <no description>
      ecdsap256sha256: <no description>
      ecdsap384sha384: <no description>
    """
        rsasha1 = 0
        rsasha256 = 1
        rsasha512 = 2
        ecdsap256sha256 = 3
        ecdsap384sha384 = 4

    class TypeValueValuesEnum(_messages.Enum):
        """One of "KEY_SIGNING" or "ZONE_SIGNING". Keys of type KEY_SIGNING have
    the Secure Entry Point flag set and, when active, are used to sign only
    resource record sets of type DNSKEY. Otherwise, the Secure Entry Point
    flag is cleared, and this key is used to sign only resource record sets of
    other types. Immutable after creation time.

    Values:
      keySigning: <no description>
      zoneSigning: <no description>
    """
        keySigning = 0
        zoneSigning = 1
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    creationTime = _messages.StringField(2)
    description = _messages.StringField(3)
    digests = _messages.MessageField('DnsKeyDigest', 4, repeated=True)
    id = _messages.StringField(5)
    isActive = _messages.BooleanField(6)
    keyLength = _messages.IntegerField(7, variant=_messages.Variant.UINT32)
    keyTag = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    kind = _messages.StringField(9, default='dns#dnsKey')
    publicKey = _messages.StringField(10)
    type = _messages.EnumField('TypeValueValuesEnum', 11)