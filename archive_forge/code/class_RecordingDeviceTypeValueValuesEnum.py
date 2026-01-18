from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecordingDeviceTypeValueValuesEnum(_messages.Enum):
    """The type of device the speech was recorded with.

    Values:
      RECORDING_DEVICE_TYPE_UNSPECIFIED: The recording device is unknown.
      SMARTPHONE: Speech was recorded on a smartphone.
      PC: Speech was recorded using a personal computer or tablet.
      PHONE_LINE: Speech was recorded over a phone line.
      VEHICLE: Speech was recorded in a vehicle.
      OTHER_OUTDOOR_DEVICE: Speech was recorded outdoors.
      OTHER_INDOOR_DEVICE: Speech was recorded indoors.
    """
    RECORDING_DEVICE_TYPE_UNSPECIFIED = 0
    SMARTPHONE = 1
    PC = 2
    PHONE_LINE = 3
    VEHICLE = 4
    OTHER_OUTDOOR_DEVICE = 5
    OTHER_INDOOR_DEVICE = 6