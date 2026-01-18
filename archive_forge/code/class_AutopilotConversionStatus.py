from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutopilotConversionStatus(_messages.Message):
    """AutopilotConversionStatus represents conversion status.

  Enums:
    StateValueValuesEnum: Output only. The current state of the conversion.
    TypeValueValuesEnum: Output only. Type represents the direction of
      conversion.

  Fields:
    autoCommitTime: Conversion will be automatically committed after this
      time.
    autopilotNodeCount: Output only. The number of Autopilot nodes in the
      cluster. This field is only updated while MIGRATING.
    standardNodeCount: Output only. The number of Standard nodes in the
      cluster. This field is only updated while MIGRATING.
    state: Output only. The current state of the conversion.
    type: Output only. Type represents the direction of conversion.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the conversion.

    Values:
      STATE_UNSPECIFIED: STATE_UNSPECIFIED indicates the state is unspecified.
      CONFIGURING: CONFIGURING indicates this cluster is being configured for
        conversion. The KCP will be restarted in the desired mode (i.e.
        Autopilot or Standard) and all workloads will be migrated to new
        nodes. If the cluster is being converted to Autopilot, CA rotation
        will also begin.
      MIGRATING: MIGRATING indicates this cluster is migrating workloads.
      MIGRATED_WAITING_FOR_COMMIT: MIGRATED_WAITING_FOR_COMMIT indicates this
        cluster has finished migrating all the workloads to Autopilot node
        pools and is waiting for the customer to commit the conversion. Once
        migration is committed, CA rotation will be completed and old node
        pools will be deleted. This action will be automatically performed 72
        hours after conversion.
      COMMITTING: COMMITTING indicates this cluster is finishing CA rotation
        by removing the old CA from the cluster and restarting the KCP.
        Additionally, old node pools will begin deletion.
      DONE: DONE indicates the conversion has been completed. Old node pools
        will continue being deleted in the background.
    """
        STATE_UNSPECIFIED = 0
        CONFIGURING = 1
        MIGRATING = 2
        MIGRATED_WAITING_FOR_COMMIT = 3
        COMMITTING = 4
        DONE = 5

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Type represents the direction of conversion.

    Values:
      TYPE_UNSPECIFIED: TYPE_UNSPECIFIED indicates the conversion type is
        unspecified.
      CONVERT_TO_AUTOPILOT: CONVERT_TO_AUTOPILOT indicates the conversion is
        from Standard to Autopilot.
      CONVERT_TO_STANDARD: CONVERT_TO_STANDARD indicates the conversion is
        from Autopilot to Standard.
    """
        TYPE_UNSPECIFIED = 0
        CONVERT_TO_AUTOPILOT = 1
        CONVERT_TO_STANDARD = 2
    autoCommitTime = _messages.StringField(1)
    autopilotNodeCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    standardNodeCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)