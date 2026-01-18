from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeverityOverride(_messages.Message):
    """Defines what action to take for a specific severity match.

  Enums:
    ActionValueValuesEnum: Required. Threat action override.
    SeverityValueValuesEnum: Required. Severity level to match.

  Fields:
    action: Required. Threat action override.
    severity: Required. Severity level to match.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. Threat action override.

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

    class SeverityValueValuesEnum(_messages.Enum):
        """Required. Severity level to match.

    Values:
      SEVERITY_UNSPECIFIED: Severity level not specified.
      INFORMATIONAL: Suspicious events that do not pose an immediate threat,
        but that are reported to call attention to deeper problems that could
        possibly exist.
      LOW: Warning-level threats that have very little impact on an
        organization's infrastructure. They usually require local or physical
        system access and may often result in victim privacy issues and
        information leakage.
      MEDIUM: Minor threats in which impact is minimized, that do not
        compromise the target or exploits that require an attacker to reside
        on the same local network as the victim, affect only non-standard
        configurations or obscure applications, or provide very limited
        access.
      HIGH: Threats that have the ability to become critical but have
        mitigating factors; for example, they may be difficult to exploit, do
        not result in elevated privileges, or do not have a large victim pool.
      CRITICAL: Serious threats, such as those that affect default
        installations of widely deployed software, result in root compromise
        of servers, and the exploit code is widely available to attackers. The
        attacker usually does not need any special authentication credentials
        or knowledge about the individual victims and the target does not need
        to be manipulated into performing any special functions.
    """
        SEVERITY_UNSPECIFIED = 0
        INFORMATIONAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)