from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CpuTemperatureInfoValueListEntry(_messages.Message):
    """A CpuTemperatureInfoValueListEntry object.

      Fields:
        label: CPU label
        temperature: Temperature in Celsius degrees.
      """
    label = _messages.StringField(1)
    temperature = _messages.IntegerField(2, variant=_messages.Variant.INT32)