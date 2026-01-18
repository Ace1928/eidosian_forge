from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointConfig(_messages.Message):
    """Endpoint config for this cluster

  Messages:
    HttpPortsValue: Output only. The map of port descriptions to URLs. Will
      only be populated if enable_http_port_access is true.

  Fields:
    enableHttpPortAccess: Optional. If true, enable http access to specific
      ports on the cluster from external sources. Defaults to false.
    httpPorts: Output only. The map of port descriptions to URLs. Will only be
      populated if enable_http_port_access is true.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HttpPortsValue(_messages.Message):
        """Output only. The map of port descriptions to URLs. Will only be
    populated if enable_http_port_access is true.

    Messages:
      AdditionalProperty: An additional property for a HttpPortsValue object.

    Fields:
      additionalProperties: Additional properties of type HttpPortsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HttpPortsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    enableHttpPortAccess = _messages.BooleanField(1)
    httpPorts = _messages.MessageField('HttpPortsValue', 2)