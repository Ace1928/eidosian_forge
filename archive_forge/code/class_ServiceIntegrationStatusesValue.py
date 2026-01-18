from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ServiceIntegrationStatusesValue(_messages.Message):
    """[Output Only] Represents the status of the service integration specs
    defined by the user in instance.serviceIntegrationSpecs.

    Messages:
      AdditionalProperty: An additional property for a
        ServiceIntegrationStatusesValue object.

    Fields:
      additionalProperties: Additional properties of type
        ServiceIntegrationStatusesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ServiceIntegrationStatusesValue object.

      Fields:
        key: Name of the additional property.
        value: A ResourceStatusServiceIntegrationStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ResourceStatusServiceIntegrationStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)