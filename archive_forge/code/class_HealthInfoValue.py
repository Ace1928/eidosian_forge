from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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