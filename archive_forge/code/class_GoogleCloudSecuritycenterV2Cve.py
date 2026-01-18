from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Cve(_messages.Message):
    """CVE stands for Common Vulnerabilities and Exposures. Information from
  the [CVE record](https://www.cve.org/ResourcesSupport/Glossary) that
  describes this vulnerability.

  Enums:
    ExploitationActivityValueValuesEnum: The exploitation activity of the
      vulnerability in the wild.
    ImpactValueValuesEnum: The potential impact of the vulnerability if it was
      to be exploited.

  Fields:
    cvssv3: Describe Common Vulnerability Scoring System specified at
      https://www.first.org/cvss/v3.1/specification-document
    exploitationActivity: The exploitation activity of the vulnerability in
      the wild.
    id: The unique identifier for the vulnerability. e.g. CVE-2021-34527
    impact: The potential impact of the vulnerability if it was to be
      exploited.
    observedInTheWild: Whether or not the vulnerability has been observed in
      the wild.
    references: Additional information about the CVE. e.g.
      https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-34527
    upstreamFixAvailable: Whether upstream fix is available for the CVE.
    zeroDay: Whether or not the vulnerability was zero day when the finding
      was published.
  """

    class ExploitationActivityValueValuesEnum(_messages.Enum):
        """The exploitation activity of the vulnerability in the wild.

    Values:
      EXPLOITATION_ACTIVITY_UNSPECIFIED: Invalid or empty value.
      WIDE: Exploitation has been reported or confirmed to widely occur.
      CONFIRMED: Limited reported or confirmed exploitation activities.
      AVAILABLE: Exploit is publicly available.
      ANTICIPATED: No known exploitation activity, but has a high potential
        for exploitation.
      NO_KNOWN: No known exploitation activity.
    """
        EXPLOITATION_ACTIVITY_UNSPECIFIED = 0
        WIDE = 1
        CONFIRMED = 2
        AVAILABLE = 3
        ANTICIPATED = 4
        NO_KNOWN = 5

    class ImpactValueValuesEnum(_messages.Enum):
        """The potential impact of the vulnerability if it was to be exploited.

    Values:
      RISK_RATING_UNSPECIFIED: Invalid or empty value.
      LOW: Exploitation would have little to no security impact.
      MEDIUM: Exploitation would enable attackers to perform activities, or
        could allow attackers to have a direct impact, but would require
        additional steps.
      HIGH: Exploitation would enable attackers to have a notable direct
        impact without needing to overcome any major mitigating factors.
      CRITICAL: Exploitation would fundamentally undermine the security of
        affected systems, enable actors to perform significant attacks with
        minimal effort, with little to no mitigating factors to overcome.
    """
        RISK_RATING_UNSPECIFIED = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4
    cvssv3 = _messages.MessageField('GoogleCloudSecuritycenterV2Cvssv3', 1)
    exploitationActivity = _messages.EnumField('ExploitationActivityValueValuesEnum', 2)
    id = _messages.StringField(3)
    impact = _messages.EnumField('ImpactValueValuesEnum', 4)
    observedInTheWild = _messages.BooleanField(5)
    references = _messages.MessageField('GoogleCloudSecuritycenterV2Reference', 6, repeated=True)
    upstreamFixAvailable = _messages.BooleanField(7)
    zeroDay = _messages.BooleanField(8)