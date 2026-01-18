from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsSecurityProfileGroupsPatchRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsSecurityProfileGroupsPatchRequest
  object.

  Fields:
    name: Immutable. Identifier. Name of the SecurityProfileGroup resource. It
      matches pattern `projects|organizations/*/locations/{location}/securityP
      rofileGroups/{security_profile_group}`.
    securityProfileGroup: A SecurityProfileGroup resource to be passed as the
      request body.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the SecurityProfileGroup resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask.
  """
    name = _messages.StringField(1, required=True)
    securityProfileGroup = _messages.MessageField('SecurityProfileGroup', 2)
    updateMask = _messages.StringField(3)