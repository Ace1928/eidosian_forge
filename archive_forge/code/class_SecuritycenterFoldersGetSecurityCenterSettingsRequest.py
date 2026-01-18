from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersGetSecurityCenterSettingsRequest(_messages.Message):
    """A SecuritycenterFoldersGetSecurityCenterSettingsRequest object.

  Fields:
    name: Required. The name of the SecurityCenterSettings to retrieve.
      Format: organizations/{organization}/securityCenterSettings Format:
      folders/{folder}/securityCenterSettings Format:
      projects/{project}/securityCenterSettings
  """
    name = _messages.StringField(1, required=True)