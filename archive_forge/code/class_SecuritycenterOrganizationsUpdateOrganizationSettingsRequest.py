from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsUpdateOrganizationSettingsRequest(_messages.Message):
    """A SecuritycenterOrganizationsUpdateOrganizationSettingsRequest object.

  Fields:
    name: The relative resource name of the settings. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Example: "organizations/{organization_id}/organizationSettings".
    organizationSettings: A OrganizationSettings resource to be passed as the
      request body.
    updateMask: The FieldMask to use when updating the settings resource. If
      empty all mutable fields will be updated.
  """
    name = _messages.StringField(1, required=True)
    organizationSettings = _messages.MessageField('OrganizationSettings', 2)
    updateMask = _messages.StringField(3)