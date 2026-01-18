from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NotificationParametersValue(_messages.Message):
    """Optional. notification_parameter are information that service
    producers may like to include that is not relevant to Rollout. This
    parameter will only be passed to Gamma and Cloud Logging for
    notification/logging purpose.

    Messages:
      AdditionalProperty: An additional property for a
        NotificationParametersValue object.

    Fields:
      additionalProperties: Additional properties of type
        NotificationParametersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NotificationParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudSaasacceleratorManagementProvidersV1NotificationParameter
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1NotificationParameter', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)