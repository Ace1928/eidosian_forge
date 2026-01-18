from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerOrganizationsLocationsCheckOnboardingStatusRequest(_messages.Message):
    """A
  PrivilegedaccessmanagerOrganizationsLocationsCheckOnboardingStatusRequest
  object.

  Fields:
    parent: Required. The resource for which the onboarding status should be
      checked. Should be in one of the following formats: *
      `projects/{project-number|project-id}/locations/{region}` *
      `folders/{folder-number}/locations/{region}` *
      `organizations/{organization-number}/locations/{region}`
  """
    parent = _messages.StringField(1, required=True)