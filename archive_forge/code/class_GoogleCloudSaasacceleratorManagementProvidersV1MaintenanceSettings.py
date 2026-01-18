from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1MaintenanceSettings(_messages.Message):
    """Maintenance settings associated with instance. Allows service producers
  and end users to assign settings that controls maintenance on this instance.

  Messages:
    MaintenancePoliciesValue: Optional. The MaintenancePolicies that have been
      attached to the instance. The key must be of the type name of the oneof
      policy name defined in MaintenancePolicy, and the embedded policy must
      define the same policy type. For details, please refer to go/mr-user-
      guide. Should not be set if maintenance_policy_names is set. If only the
      name is needed, then only populate MaintenancePolicy.name.

  Fields:
    exclude: Optional. Exclude instance from maintenance. When true, rollout
      service will not attempt maintenance on the instance. Rollout service
      will include the instance in reported rollout progress as not attempted.
    isRollback: Optional. If the update call is triggered from rollback, set
      the value as true.
    maintenancePolicies: Optional. The MaintenancePolicies that have been
      attached to the instance. The key must be of the type name of the oneof
      policy name defined in MaintenancePolicy, and the embedded policy must
      define the same policy type. For details, please refer to go/mr-user-
      guide. Should not be set if maintenance_policy_names is set. If only the
      name is needed, then only populate MaintenancePolicy.name.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MaintenancePoliciesValue(_messages.Message):
        """Optional. The MaintenancePolicies that have been attached to the
    instance. The key must be of the type name of the oneof policy name
    defined in MaintenancePolicy, and the embedded policy must define the same
    policy type. For details, please refer to go/mr-user-guide. Should not be
    set if maintenance_policy_names is set. If only the name is needed, then
    only populate MaintenancePolicy.name.

    Messages:
      AdditionalProperty: An additional property for a
        MaintenancePoliciesValue object.

    Fields:
      additionalProperties: Additional properties of type
        MaintenancePoliciesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MaintenancePoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A MaintenancePolicy attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('MaintenancePolicy', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    exclude = _messages.BooleanField(1)
    isRollback = _messages.BooleanField(2)
    maintenancePolicies = _messages.MessageField('MaintenancePoliciesValue', 3)