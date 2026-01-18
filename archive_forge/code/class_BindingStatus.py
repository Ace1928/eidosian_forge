from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BindingStatus(_messages.Message):
    """The binding status of a resource

  Messages:
    AnnotationsValue: Annotations of the Cloud Run service for the binded
      resource.
    EnvironmentVariablesValue: Environment variables of the Cloud Run service
      for the binded resource.

  Fields:
    annotations: Annotations of the Cloud Run service for the binded resource.
    environmentVariables: Environment variables of the Cloud Run service for
      the binded resource.
    resourceName: Name of the binded resource.
    resourceType: Type of the binded resource.
    serviceAccount: Service account email used by the Cloud Run service for
      the binded resource.
    serviceName: Name of the Cloud Run service.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations of the Cloud Run service for the binded resource.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentVariablesValue(_messages.Message):
        """Environment variables of the Cloud Run service for the binded
    resource.

    Messages:
      AdditionalProperty: An additional property for a
        EnvironmentVariablesValue object.

    Fields:
      additionalProperties: Additional properties of type
        EnvironmentVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvironmentVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    environmentVariables = _messages.MessageField('EnvironmentVariablesValue', 2)
    resourceName = _messages.StringField(3)
    resourceType = _messages.StringField(4)
    serviceAccount = _messages.StringField(5)
    serviceName = _messages.StringField(6)