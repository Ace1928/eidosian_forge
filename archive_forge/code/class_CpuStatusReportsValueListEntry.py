from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CpuStatusReportsValueListEntry(_messages.Message):
    """A CpuStatusReportsValueListEntry object.

    Messages:
      CpuTemperatureInfoValueListEntry: A CpuTemperatureInfoValueListEntry
        object.

    Fields:
      cpuTemperatureInfo: List of CPU temperature samples.
      cpuUtilizationPercentageInfo: A integer attribute.
      reportTime: Date and time the report was received.
    """

    class CpuTemperatureInfoValueListEntry(_messages.Message):
        """A CpuTemperatureInfoValueListEntry object.

      Fields:
        label: CPU label
        temperature: Temperature in Celsius degrees.
      """
        label = _messages.StringField(1)
        temperature = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    cpuTemperatureInfo = _messages.MessageField('CpuTemperatureInfoValueListEntry', 1, repeated=True)
    cpuUtilizationPercentageInfo = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)
    reportTime = _message_types.DateTimeField(3)