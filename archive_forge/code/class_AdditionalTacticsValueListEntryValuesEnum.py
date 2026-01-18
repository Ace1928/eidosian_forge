from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdditionalTacticsValueListEntryValuesEnum(_messages.Enum):
    """AdditionalTacticsValueListEntryValuesEnum enum type.

    Values:
      TACTIC_UNSPECIFIED: Unspecified value.
      RECONNAISSANCE: TA0043
      RESOURCE_DEVELOPMENT: TA0042
      INITIAL_ACCESS: TA0001
      EXECUTION: TA0002
      PERSISTENCE: TA0003
      PRIVILEGE_ESCALATION: TA0004
      DEFENSE_EVASION: TA0005
      CREDENTIAL_ACCESS: TA0006
      DISCOVERY: TA0007
      LATERAL_MOVEMENT: TA0008
      COLLECTION: TA0009
      COMMAND_AND_CONTROL: TA0011
      EXFILTRATION: TA0010
      IMPACT: TA0040
    """
    TACTIC_UNSPECIFIED = 0
    RECONNAISSANCE = 1
    RESOURCE_DEVELOPMENT = 2
    INITIAL_ACCESS = 3
    EXECUTION = 4
    PERSISTENCE = 5
    PRIVILEGE_ESCALATION = 6
    DEFENSE_EVASION = 7
    CREDENTIAL_ACCESS = 8
    DISCOVERY = 9
    LATERAL_MOVEMENT = 10
    COLLECTION = 11
    COMMAND_AND_CONTROL = 12
    EXFILTRATION = 13
    IMPACT = 14