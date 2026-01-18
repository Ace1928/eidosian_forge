from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Account(_messages.Message):
    """Represents an account that was established by the customer with a
  service provider. When consuming services on external service provider's
  systems, the service provider generally needs to create a linked-account on
  their system to track customers. The account resource represents this
  relationship. Products/Services that are hosted by external service
  providers generally require an account to be present before they can be
  purchased and used. The metadata that indicates whether an Account is
  required for a purchase, or what parameters are needed for creating an
  Account is configured by service providers.

  Messages:
    PropertiesValue: Output only. Set of properties that the service provider
      supplied during account creation.

  Fields:
    approvals: Output only. The approvals for this account. These approvals
      are used to track actions that are permitted or have been completed by a
      customer within the context of the provider. This might include a sign
      up flow or a provisioning step, for example, that the provider can admit
      to having happened.
    createTime: Output only. The creation timestamp.
    name: Output only. The resource name of the account. Account names have
      the form `billingAccounts/{billing_account_id}/accounts/{account_id}`.
    properties: Output only. Set of properties that the service provider
      supplied during account creation.
    provider: Required. The identifier of the service provider that this
      account was created against. Provider has the format of
      `providers/{provider_id}`.
    updateTime: Output only. The last update timestamp.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Output only. Set of properties that the service provider supplied
    during account creation.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    approvals = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1AccountApproval', 1, repeated=True)
    createTime = _messages.StringField(2)
    name = _messages.StringField(3)
    properties = _messages.MessageField('PropertiesValue', 4)
    provider = _messages.StringField(5)
    updateTime = _messages.StringField(6)