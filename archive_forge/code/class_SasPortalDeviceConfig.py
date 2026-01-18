from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeviceConfig(_messages.Message):
    """Information about the device configuration.

  Enums:
    CategoryValueValuesEnum: FCC category of the device.
    MeasurementCapabilitiesValueListEntryValuesEnum:
    StateValueValuesEnum: State of the configuration.

  Fields:
    airInterface: Information about this device's air interface.
    callSign: The call sign of the device operator.
    category: FCC category of the device.
    installationParams: Installation parameters for the device.
    isSigned: Output only. Whether the configuration has been signed by a CPI.
    measurementCapabilities: Measurement reporting capabilities of the device.
    model: Information about this device model.
    state: State of the configuration.
    updateTime: Output only. The last time the device configuration was
      edited.
    userId: The identifier of a device user.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """FCC category of the device.

    Values:
      DEVICE_CATEGORY_UNSPECIFIED: Unspecified device category.
      DEVICE_CATEGORY_A: Category A.
      DEVICE_CATEGORY_B: Category B.
    """
        DEVICE_CATEGORY_UNSPECIFIED = 0
        DEVICE_CATEGORY_A = 1
        DEVICE_CATEGORY_B = 2

    class MeasurementCapabilitiesValueListEntryValuesEnum(_messages.Enum):
        """MeasurementCapabilitiesValueListEntryValuesEnum enum type.

    Values:
      MEASUREMENT_CAPABILITY_UNSPECIFIED: <no description>
      MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITH_GRANT: <no description>
      MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITHOUT_GRANT: <no description>
    """
        MEASUREMENT_CAPABILITY_UNSPECIFIED = 0
        MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITH_GRANT = 1
        MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITHOUT_GRANT = 2

    class StateValueValuesEnum(_messages.Enum):
        """State of the configuration.

    Values:
      DEVICE_CONFIG_STATE_UNSPECIFIED: <no description>
      DRAFT: <no description>
      FINAL: <no description>
    """
        DEVICE_CONFIG_STATE_UNSPECIFIED = 0
        DRAFT = 1
        FINAL = 2
    airInterface = _messages.MessageField('SasPortalDeviceAirInterface', 1)
    callSign = _messages.StringField(2)
    category = _messages.EnumField('CategoryValueValuesEnum', 3)
    installationParams = _messages.MessageField('SasPortalInstallationParams', 4)
    isSigned = _messages.BooleanField(5)
    measurementCapabilities = _messages.EnumField('MeasurementCapabilitiesValueListEntryValuesEnum', 6, repeated=True)
    model = _messages.MessageField('SasPortalDeviceModel', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)
    userId = _messages.StringField(10)