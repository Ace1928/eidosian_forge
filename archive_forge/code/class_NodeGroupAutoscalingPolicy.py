from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupAutoscalingPolicy(_messages.Message):
    """A NodeGroupAutoscalingPolicy object.

  Enums:
    ModeValueValuesEnum: The autoscaling mode. Set to one of: ON, OFF, or
      ONLY_SCALE_OUT. For more information, see Autoscaler modes.

  Fields:
    maxNodes: The maximum number of nodes that the group should have. Must be
      set if autoscaling is enabled. Maximum value allowed is 100.
    minNodes: The minimum number of nodes that the group should have.
    mode: The autoscaling mode. Set to one of: ON, OFF, or ONLY_SCALE_OUT. For
      more information, see Autoscaler modes.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """The autoscaling mode. Set to one of: ON, OFF, or ONLY_SCALE_OUT. For
    more information, see Autoscaler modes.

    Values:
      MODE_UNSPECIFIED: <no description>
      OFF: Autoscaling is disabled.
      ON: Autocaling is fully enabled.
      ONLY_SCALE_OUT: Autoscaling will only scale out and will not remove
        nodes.
    """
        MODE_UNSPECIFIED = 0
        OFF = 1
        ON = 2
        ONLY_SCALE_OUT = 3
    maxNodes = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minNodes = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    mode = _messages.EnumField('ModeValueValuesEnum', 3)