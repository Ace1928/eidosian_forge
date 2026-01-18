from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceGroupHealth(_messages.Message):
    """A BackendServiceGroupHealth object.

  Messages:
    AnnotationsValue: Metadata defined as annotations on the network endpoint
      group.

  Fields:
    annotations: Metadata defined as annotations on the network endpoint
      group.
    healthStatus: Health state of the backend instances or endpoints in
      requested instance or network endpoint group, determined based on
      configured health checks.
    kind: [Output Only] Type of resource. Always
      compute#backendServiceGroupHealth for the health of backend services.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Metadata defined as annotations on the network endpoint group.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    healthStatus = _messages.MessageField('HealthStatus', 2, repeated=True)
    kind = _messages.StringField(3, default='compute#backendServiceGroupHealth')