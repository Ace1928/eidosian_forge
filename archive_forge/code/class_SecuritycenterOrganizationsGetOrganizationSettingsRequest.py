from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsGetOrganizationSettingsRequest(_messages.Message):
    """A SecuritycenterOrganizationsGetOrganizationSettingsRequest object.

  Fields:
    name: Required. Name of the organization to get organization settings for.
      Its format is "organizations/[organization_id]/organizationSettings".
  """
    name = _messages.StringField(1, required=True)