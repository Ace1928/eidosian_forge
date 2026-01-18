from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneDnsSecConfig(_messages.Message):
    """A ManagedZoneDnsSecConfig object.

  Enums:
    NonExistenceValueValuesEnum: Specifies the mechanism for authenticated
      denial-of-existence responses. Can only be changed while the state is
      OFF.
    StateValueValuesEnum: Specifies whether DNSSEC is enabled, and what mode
      it is in.

  Fields:
    defaultKeySpecs: Specifies parameters for generating initial DnsKeys for
      this ManagedZone. Can only be changed while the state is OFF.
    kind: A string attribute.
    nonExistence: Specifies the mechanism for authenticated denial-of-
      existence responses. Can only be changed while the state is OFF.
    state: Specifies whether DNSSEC is enabled, and what mode it is in.
  """

    class NonExistenceValueValuesEnum(_messages.Enum):
        """Specifies the mechanism for authenticated denial-of-existence
    responses. Can only be changed while the state is OFF.

    Values:
      nsec: <no description>
      nsec3: <no description>
    """
        nsec = 0
        nsec3 = 1

    class StateValueValuesEnum(_messages.Enum):
        """Specifies whether DNSSEC is enabled, and what mode it is in.

    Values:
      off: DNSSEC is disabled; the zone is not signed.
      on: DNSSEC is enabled; the zone is signed and fully managed.
      transfer: DNSSEC is enabled, but in a "transfer" mode.
    """
        off = 0
        on = 1
        transfer = 2
    defaultKeySpecs = _messages.MessageField('DnsKeySpec', 1, repeated=True)
    kind = _messages.StringField(2, default='dns#managedZoneDnsSecConfig')
    nonExistence = _messages.EnumField('NonExistenceValueValuesEnum', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)