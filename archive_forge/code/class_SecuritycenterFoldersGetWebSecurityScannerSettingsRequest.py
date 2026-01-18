from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersGetWebSecurityScannerSettingsRequest(_messages.Message):
    """A SecuritycenterFoldersGetWebSecurityScannerSettingsRequest object.

  Fields:
    name: Required. The name of the WebSecurityScannerSettings to retrieve.
      Formats: * organizations/{organization}/webSecurityScannerSettings *
      folders/{folder}/webSecurityScannerSettings *
      projects/{project}/webSecurityScannerSettings
  """
    name = _messages.StringField(1, required=True)