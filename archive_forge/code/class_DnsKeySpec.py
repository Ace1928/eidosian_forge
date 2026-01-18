from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsKeySpec(_messages.Message):
    """Parameters for DnsKey key generation. Used for generating initial keys
  for a new ManagedZone and as default when adding a new DnsKey.

  Enums:
    AlgorithmValueValuesEnum: String mnemonic specifying the DNSSEC algorithm
      of this key.
    KeyTypeValueValuesEnum: Specifies whether this is a key signing key (KSK)
      or a zone signing key (ZSK). Key signing keys have the Secure Entry
      Point flag set and, when active, are only used to sign resource record
      sets of type DNSKEY. Zone signing keys do not have the Secure Entry
      Point flag set and are used to sign all other types of resource record
      sets.

  Fields:
    algorithm: String mnemonic specifying the DNSSEC algorithm of this key.
    keyLength: Length of the keys in bits.
    keyType: Specifies whether this is a key signing key (KSK) or a zone
      signing key (ZSK). Key signing keys have the Secure Entry Point flag set
      and, when active, are only used to sign resource record sets of type
      DNSKEY. Zone signing keys do not have the Secure Entry Point flag set
      and are used to sign all other types of resource record sets.
    kind: A string attribute.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """String mnemonic specifying the DNSSEC algorithm of this key.

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

    class KeyTypeValueValuesEnum(_messages.Enum):
        """Specifies whether this is a key signing key (KSK) or a zone signing
    key (ZSK). Key signing keys have the Secure Entry Point flag set and, when
    active, are only used to sign resource record sets of type DNSKEY. Zone
    signing keys do not have the Secure Entry Point flag set and are used to
    sign all other types of resource record sets.

    Values:
      keySigning: <no description>
      zoneSigning: <no description>
    """
        keySigning = 0
        zoneSigning = 1
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    keyLength = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    keyType = _messages.EnumField('KeyTypeValueValuesEnum', 3)
    kind = _messages.StringField(4, default='dns#dnsKeySpec')