from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrglifecycleOrganizationsLocationsManagedOrganizationsCreateRequest(_messages.Message):
    """A OrglifecycleOrganizationsLocationsManagedOrganizationsCreateRequest
  object.

  Fields:
    managedOrganization: A ManagedOrganization resource to be passed as the
      request body.
    managedOrganizationId: Required. User specified Managed Organization ID.
      This has to be unique under parent:
      organizations/{organization_id}/locations/{location} It must be 6 to 30
      lowercase ASCII letters, digits, or hyphens. It must start with a
      letter.Trailing hyphens are prohibited. Example: tokyo-rain-123
    parent: Required. The parent resource where this ManagedOrganization will
      be created. Must be in the format:
      organizations/{organization_id}/locations/{location}.
  """
    managedOrganization = _messages.MessageField('ManagedOrganization', 1)
    managedOrganizationId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)