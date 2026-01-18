from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CPUInfo(_messages.Message):
    """A CPUInfo object.

  Fields:
    cpuProcessor: description of the device processor ie '1.8 GHz hexa core
      64-bit ARMv8-A'
    cpuSpeedInGhz: the CPU clock speed in GHz
    numberOfCores: the number of CPU cores
  """
    cpuProcessor = _messages.StringField(1)
    cpuSpeedInGhz = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    numberOfCores = _messages.IntegerField(3, variant=_messages.Variant.INT32)