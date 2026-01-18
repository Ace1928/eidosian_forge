from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MaintenancePolicyNamesValue(_messages.Message):
    """Optional. The MaintenancePolicies that have been attached to the
    instance. The key must be of the type name of the oneof policy name
    defined in MaintenancePolicy, and the referenced policy must define the
    same policy type. For details, please refer to go/mr-user-guide. Should
    not be set if maintenance_settings.maintenance_policies is set.

    Messages:
      AdditionalProperty: An additional property for a
        MaintenancePolicyNamesValue object.

    Fields:
      additionalProperties: Additional properties of type
        MaintenancePolicyNamesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MaintenancePolicyNamesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)