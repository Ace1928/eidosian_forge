from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MqttConfig(_messages.Message):
    """The configuration of MQTT for a device registry.

  Enums:
    MqttEnabledStateValueValuesEnum: If enabled, allows connections using the
      MQTT protocol. Otherwise, MQTT connections to this registry will fail.

  Fields:
    mqttEnabledState: If enabled, allows connections using the MQTT protocol.
      Otherwise, MQTT connections to this registry will fail.
  """

    class MqttEnabledStateValueValuesEnum(_messages.Enum):
        """If enabled, allows connections using the MQTT protocol. Otherwise,
    MQTT connections to this registry will fail.

    Values:
      MQTT_STATE_UNSPECIFIED: No MQTT state specified. If not specified, MQTT
        will be enabled by default.
      MQTT_ENABLED: Enables a MQTT connection.
      MQTT_DISABLED: Disables a MQTT connection.
    """
        MQTT_STATE_UNSPECIFIED = 0
        MQTT_ENABLED = 1
        MQTT_DISABLED = 2
    mqttEnabledState = _messages.EnumField('MqttEnabledStateValueValuesEnum', 1)