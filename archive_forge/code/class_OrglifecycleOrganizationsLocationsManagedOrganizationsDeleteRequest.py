from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrglifecycleOrganizationsLocationsManagedOrganizationsDeleteRequest(_messages.Message):
    """A OrglifecycleOrganizationsLocationsManagedOrganizationsDeleteRequest
  object.

  Fields:
    name: Required. The name of the ManagedOrganization to delete. Format: org
      anizations/{organization_id}/locations/*/managedOrganizations/{managed_o
      rganization_id}
  """
    name = _messages.StringField(1, required=True)