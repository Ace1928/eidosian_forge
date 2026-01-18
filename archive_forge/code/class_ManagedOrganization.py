from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedOrganization(_messages.Message):
    """This organization managed by GCP resellers.

  Enums:
    StateValueValuesEnum: Output only. The state of the managed organization
      and cloudresourcemanager.googleapis.com/Organization created on behalf
      of the customer.

  Fields:
    admins: Optional. List of organization admins.
    createTime: Output only. The timestamp for the managed organization was
      created.
    deleteTime: Output only. The timestamp that the managed organization was
      soft deleted.
    name: Identifier. The resource name of the managed organization. Format: o
      rganizations/{organization_id}/locations/{location}/managedOrganizations
      /{managed_organization_id}
    organizationDisplayName: Required. Immutable. The display name of the
      cloudresourcemanager.googleapis.com/Organization created on behalf of
      the customer.
    organizationNumber: Output only. System generated ID for the
      cloudresourcemanager.googleapis.com/Organization created on behalf of
      the customer.
    purgeTime: Output only. Time after which the managed organization will be
      permanently purged and cannot be recovered.
    state: Output only. The state of the managed organization and
      cloudresourcemanager.googleapis.com/Organization created on behalf of
      the customer.
    updateTime: Output only. The timestamp for the last update of the managed
      organization.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the managed organization and
    cloudresourcemanager.googleapis.com/Organization created on behalf of the
    customer.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The organization of the
        cloudresourcemanager.googleapis.com/Organization created on behalf of
        the customer is soft-deleted.
      DELETED: The organization of the
        cloudresourcemanager.googleapis.com/Organization created on behalf of
        the customer is soft-deleted. Soft-deleted organization are
        permanently deleted after approximately 30 days. You can restore a
        soft-deleted organization using
        [Orglifecycle.UndeleteManagedOrganization]. You cannot reuse the ID of
        a soft-deleted organization until it is permanently deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    admins = _messages.MessageField('OrganizationAdmin', 1, repeated=True)
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    name = _messages.StringField(4)
    organizationDisplayName = _messages.StringField(5)
    organizationNumber = _messages.IntegerField(6)
    purgeTime = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)