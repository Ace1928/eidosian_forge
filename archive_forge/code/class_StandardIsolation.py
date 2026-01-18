from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardIsolation(_messages.Message):
    """Standard options for isolating this app profile's traffic from other use
  cases.

  Enums:
    PriorityValueValuesEnum: The priority of requests sent using this app
      profile.

  Fields:
    priority: The priority of requests sent using this app profile.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """The priority of requests sent using this app profile.

    Values:
      PRIORITY_UNSPECIFIED: Default value. Mapped to PRIORITY_HIGH (the legacy
        behavior) on creation.
      PRIORITY_LOW: <no description>
      PRIORITY_MEDIUM: <no description>
      PRIORITY_HIGH: <no description>
    """
        PRIORITY_UNSPECIFIED = 0
        PRIORITY_LOW = 1
        PRIORITY_MEDIUM = 2
        PRIORITY_HIGH = 3
    priority = _messages.EnumField('PriorityValueValuesEnum', 1)