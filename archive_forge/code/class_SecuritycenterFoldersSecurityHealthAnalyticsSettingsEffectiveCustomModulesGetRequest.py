from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersSecurityHealthAnalyticsSettingsEffectiveCustomModulesGetRequest(_messages.Message):
    """A SecuritycenterFoldersSecurityHealthAnalyticsSettingsEffectiveCustomMod
  ulesGetRequest object.

  Fields:
    name: Required. Name of the effective custom module to get. Its format is
      "organizations/{organization}/securityHealthAnalyticsSettings/effectiveC
      ustomModules/{customModule}", "folders/{folder}/securityHealthAnalyticsS
      ettings/effectiveCustomModules/{customModule}", or "projects/{project}/s
      ecurityHealthAnalyticsSettings/effectiveCustomModules/{customModule}"
  """
    name = _messages.StringField(1, required=True)