from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UtilizationReport(_messages.Message):
    """Utilization report details the utilization (CPU, memory, etc.) of
  selected source VMs.

  Enums:
    StateValueValuesEnum: Output only. Current state of the report.
    TimeFrameValueValuesEnum: Time frame of the report.

  Fields:
    createTime: Output only. The time the report was created (this refers to
      the time of the request, not the time the report creation completed).
    displayName: The report display name, as assigned by the user.
    error: Output only. Provides details on the state of the report in case of
      an error.
    frameEndTime: Output only. The point in time when the time frame ends.
      Notice that the time frame is counted backwards. For instance if the
      "frame_end_time" value is 2021/01/20 and the time frame is WEEK then the
      report covers the week between 2021/01/20 and 2021/01/14.
    name: Output only. The report unique name.
    state: Output only. Current state of the report.
    stateTime: Output only. The time the state was last set.
    timeFrame: Time frame of the report.
    vmCount: Output only. Total number of VMs included in the report.
    vms: List of utilization information per VM. When sent as part of the
      request, the "vm_id" field is used in order to specify which VMs to
      include in the report. In that case all other fields are ignored.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the report.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This value is not in use.
      CREATING: The report is in the making.
      SUCCEEDED: Report creation completed successfully.
      FAILED: Report creation failed.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        SUCCEEDED = 2
        FAILED = 3

    class TimeFrameValueValuesEnum(_messages.Enum):
        """Time frame of the report.

    Values:
      TIME_FRAME_UNSPECIFIED: The time frame was not specified and will
        default to WEEK.
      WEEK: One week.
      MONTH: One month.
      YEAR: One year.
    """
        TIME_FRAME_UNSPECIFIED = 0
        WEEK = 1
        MONTH = 2
        YEAR = 3
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    error = _messages.MessageField('Status', 3)
    frameEndTime = _messages.StringField(4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stateTime = _messages.StringField(7)
    timeFrame = _messages.EnumField('TimeFrameValueValuesEnum', 8)
    vmCount = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    vms = _messages.MessageField('VmUtilizationInfo', 10, repeated=True)