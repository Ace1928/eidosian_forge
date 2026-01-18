from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersSecurityHealthAnalyticsSettingsCalculateRequest(_messages.Message):
    """A SecuritycenterFoldersSecurityHealthAnalyticsSettingsCalculateRequest
  object.

  Fields:
    name: Required. The name of the SecurityHealthAnalyticsSettings to
      calculate. Formats: *
      organizations/{organization}/securityHealthAnalyticsSettings *
      folders/{folder}/securityHealthAnalyticsSettings *
      projects/{project}/securityHealthAnalyticsSettings
  """
    name = _messages.StringField(1, required=True)