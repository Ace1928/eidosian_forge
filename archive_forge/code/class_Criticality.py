from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Criticality(_messages.Message):
    """Criticality of the Application, Service, or Workload

  Enums:
    TypeValueValuesEnum: Required. Criticality Type.

  Fields:
    level: Optional. Criticality level. Can contain only lowercase letters,
      numeric characters, underscores, and dashes. Can have a maximum length
      of 63 characters. Deprecated: Please refer to type instead.
    missionCritical: Optional. Indicates mission-critical Application,
      Service, or Workload. Deprecated: Please refer to type instead.
    type: Required. Criticality Type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Criticality Type.

    Values:
      TYPE_UNSPECIFIED: Unspecified type.
      MISSION_CRITICAL: Mission critical service, application or workload.
      HIGH: High impact.
      MEDIUM: Medium impact.
      LOW: Low impact.
    """
        TYPE_UNSPECIFIED = 0
        MISSION_CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4
    level = _messages.StringField(1)
    missionCritical = _messages.BooleanField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)