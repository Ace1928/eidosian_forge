from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrglifecycleOrganizationsLocationsManagedOrganizationsPatchRequest(_messages.Message):
    """A OrglifecycleOrganizationsLocationsManagedOrganizationsPatchRequest
  object.

  Fields:
    managedOrganization: A ManagedOrganization resource to be passed as the
      request body.
    name: Identifier. The resource name of the managed organization. Format: o
      rganizations/{organization_id}/locations/{location}/managedOrganizations
      /{managed_organization_id}
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the ManagedOrganization resource by the update. The list
      of fields to update. Supported field: ManagedOrganization.admins;
      Provided admin list will replace previous list.
  """
    managedOrganization = _messages.MessageField('ManagedOrganization', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)