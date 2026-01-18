from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsUpdateWebSecurityScannerSettingsRequest(_messages.Message):
    """A SecuritycenterOrganizationsUpdateWebSecurityScannerSettingsRequest
  object.

  Fields:
    name: The resource name of the WebSecurityScannerSettings. Formats: *
      organizations/{organization}/webSecurityScannerSettings *
      folders/{folder}/webSecurityScannerSettings *
      projects/{project}/webSecurityScannerSettings
    updateMask: The list of fields to be updated.
    webSecurityScannerSettings: A WebSecurityScannerSettings resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    webSecurityScannerSettings = _messages.MessageField('WebSecurityScannerSettings', 3)