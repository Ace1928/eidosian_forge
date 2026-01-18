from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Entitlement(_messages.Message):
    """Entitlement represents the ability to use a product or services
  associated with a purchase within a Project. When the customer creates an
  Order, the system will create Entitlement resources under projects
  associated with the same billing account as the order, for all
  products/services procured in the order. Users can enable/disable
  Entitlements to allow/disallow using the product/service in a project. Next
  Id: 26

  Enums:
    StateValueValuesEnum: Output only. The state of the entitlement.

  Fields:
    changeHistory: Output only. Changes that are not pending anymore, e.g. it
      was effective at some point, or the change was reverted by the customer,
      or the change was rejected by partner. No more operations are allowed on
      these changes.
    createTime: Output only. The create timestamp.
    flavorExternalName: Output only. External name of the flavor this
      entitlement is created against. This field is populated when entitlement
      has currently associated flavor, it is empty when entitlement is
      UNAVAILABLE (if order is pending activation or order is already
      cancelled).
    name: Output only. The resource Name of the Entitlement. Entitlement names
      have the form `projects/{project_id}/entitlements/{entitlement_id}`.
    order: Output only. Order associated with this Entitlement. In the format
      of `billingAccounts/{billing_account}/orders/{order}`
    pendingChange: Output only. A change which is pending and not yet
      effective.
    productExternalName: Output only. External name of the product this
      entitlement is created against.
    provider: Output only. Provider associated with this Entitlement. In the
      format of `providers/{provider_id}`.
    state: Output only. The state of the entitlement.
    stateReason: Output only. An explanation for the entitlement's state.
      Mainly used in the case of
      `EntitlementState.ENTITLEMENT_STATE_UNAVAILABLE` states to explain why
      the entitlement is unavailable.
    updateTime: Output only. The last update timestamp.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the entitlement.

    Values:
      ENTITLEMENT_STATE_UNSPECIFIED: Sentinel value. Do not use.
      ENTITLEMENT_STATE_UNAVAILABLE: Indicates that the entitlement is
        unavailable and cannot be enabled.
      ENTITLEMENT_STATE_ENABLED: Indicates that the entitlement is enabled.
        The procured item is now usable.
      ENTITLEMENT_STATE_DISABLED: Indicates that the entitlement is disabled.
        The procured item is not usable.
      ENTITLEMENT_STATE_EXHAUSTED: Indicates that no more procured products
        can be added to the current project. This will be returned if there is
        already a consumer entitlement with resources deployed in another
        project and the product allows a single deployment only.
      ENTITLEMENT_STATE_INELIGIBLE: Indicates that the entitlement is
        ineligible for usage because the project is already enabled as a
        consumer on another entitlement of the same product.
    """
        ENTITLEMENT_STATE_UNSPECIFIED = 0
        ENTITLEMENT_STATE_UNAVAILABLE = 1
        ENTITLEMENT_STATE_ENABLED = 2
        ENTITLEMENT_STATE_DISABLED = 3
        ENTITLEMENT_STATE_EXHAUSTED = 4
        ENTITLEMENT_STATE_INELIGIBLE = 5
    changeHistory = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1EntitlementChange', 1, repeated=True)
    createTime = _messages.StringField(2)
    flavorExternalName = _messages.StringField(3)
    name = _messages.StringField(4)
    order = _messages.StringField(5)
    pendingChange = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1EntitlementChange', 6)
    productExternalName = _messages.StringField(7)
    provider = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    stateReason = _messages.StringField(10)
    updateTime = _messages.StringField(11)