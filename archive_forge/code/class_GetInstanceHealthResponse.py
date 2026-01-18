from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetInstanceHealthResponse(_messages.Message):
    """Response for checking if a notebook instance is healthy.

  Enums:
    HealthStateValueValuesEnum: Output only. Runtime health_state.

  Messages:
    HealthInfoValue: Output only. Additional information about instance
      health. Example: healthInfo": { "docker_proxy_agent_status": "1",
      "docker_status": "1", "jupyterlab_api_status": "-1",
      "jupyterlab_status": "-1", "updated": "2020-10-18 09:40:03.573409" }

  Fields:
    healthInfo: Output only. Additional information about instance health.
      Example: healthInfo": { "docker_proxy_agent_status": "1",
      "docker_status": "1", "jupyterlab_api_status": "-1",
      "jupyterlab_status": "-1", "updated": "2020-10-18 09:40:03.573409" }
    healthState: Output only. Runtime health_state.
  """

    class HealthStateValueValuesEnum(_messages.Enum):
        """Output only. Runtime health_state.

    Values:
      HEALTH_STATE_UNSPECIFIED: The instance substate is unknown.
      HEALTHY: The instance is known to be in an healthy state (for example,
        critical daemons are running) Applies to ACTIVE state.
      UNHEALTHY: The instance is known to be in an unhealthy state (for
        example, critical daemons are not running) Applies to ACTIVE state.
      AGENT_NOT_INSTALLED: The instance has not installed health monitoring
        agent. Applies to ACTIVE state.
      AGENT_NOT_RUNNING: The instance health monitoring agent is not running.
        Applies to ACTIVE state.
    """
        HEALTH_STATE_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2
        AGENT_NOT_INSTALLED = 3
        AGENT_NOT_RUNNING = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HealthInfoValue(_messages.Message):
        """Output only. Additional information about instance health. Example:
    healthInfo": { "docker_proxy_agent_status": "1", "docker_status": "1",
    "jupyterlab_api_status": "-1", "jupyterlab_status": "-1", "updated":
    "2020-10-18 09:40:03.573409" }

    Messages:
      AdditionalProperty: An additional property for a HealthInfoValue object.

    Fields:
      additionalProperties: Additional properties of type HealthInfoValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HealthInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    healthInfo = _messages.MessageField('HealthInfoValue', 1)
    healthState = _messages.EnumField('HealthStateValueValuesEnum', 2)