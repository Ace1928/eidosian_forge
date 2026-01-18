from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceHAPolicy(_messages.Message):
    """A BackendServiceHAPolicy object.

  Enums:
    FastIPMoveValueValuesEnum: Enabling fastIPMove is not supported.

  Fields:
    fastIPMove: Enabling fastIPMove is not supported.
    leader: Setting a leader is not supported.
  """

    class FastIPMoveValueValuesEnum(_messages.Enum):
        """Enabling fastIPMove is not supported.

    Values:
      DISABLED: <no description>
      GARP_RA: <no description>
    """
        DISABLED = 0
        GARP_RA = 1
    fastIPMove = _messages.EnumField('FastIPMoveValueValuesEnum', 1)
    leader = _messages.MessageField('BackendServiceHAPolicyLeader', 2)