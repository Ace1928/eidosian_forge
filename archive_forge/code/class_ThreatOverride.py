from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ThreatOverride(_messages.Message):
    """Defines what action to take for a specific threat_id match.

  Enums:
    ActionValueValuesEnum: Required. Threat action override. For some threat
      types, only a subset of actions applies.
    TypeValueValuesEnum: Output only. Type of the threat (read only).

  Fields:
    action: Required. Threat action override. For some threat types, only a
      subset of actions applies.
    threatId: Required. Vendor-specific ID of a threat to override.
    type: Output only. Type of the threat (read only).
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. Threat action override. For some threat types, only a subset
    of actions applies.

    Values:
      THREAT_ACTION_UNSPECIFIED: Threat action not specified.
      DEFAULT_ACTION: The default action (as specified by the vendor) is
        taken.
      ALLOW: The packet matching this rule will be allowed to transmit.
      ALERT: The packet matching this rule will be allowed to transmit, but a
        threat_log entry will be sent to the consumer project.
      DENY: The packet matching this rule will be dropped, and a threat_log
        entry will be sent to the consumer project.
    """
        THREAT_ACTION_UNSPECIFIED = 0
        DEFAULT_ACTION = 1
        ALLOW = 2
        ALERT = 3
        DENY = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Type of the threat (read only).

    Values:
      THREAT_TYPE_UNSPECIFIED: Type of threat not specified.
      UNKNOWN: Type of threat is not derivable from threat ID. An override
        will be created for all types. Firewall will ignore overridden
        signature ID's that don't exist in the specific type.
      VULNERABILITY: Threats related to system flaws that an attacker might
        otherwise attempt to exploit.
      ANTIVIRUS: Threats related to viruses and malware found in executables
        and file types.
      SPYWARE: Threats related to command-and-control (C2) activity, where
        spyware on an infected client is collecting data without the user's
        consent and/or communicating with a remote attacker.
      DNS: Threats related to DNS.
    """
        THREAT_TYPE_UNSPECIFIED = 0
        UNKNOWN = 1
        VULNERABILITY = 2
        ANTIVIRUS = 3
        SPYWARE = 4
        DNS = 5
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    threatId = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)