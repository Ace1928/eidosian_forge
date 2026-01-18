from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersSecurityHealthAnalyticsSettingsCustomModulesCreateRequest(_messages.Message):
    """A SecuritycenterFoldersSecurityHealthAnalyticsSettingsCustomModulesCreat
  eRequest object.

  Fields:
    googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule: A
      GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule resource
      to be passed as the request body.
    parent: Required. Resource name of the new custom module's parent. Its
      format is
      "organizations/{organization}/securityHealthAnalyticsSettings",
      "folders/{folder}/securityHealthAnalyticsSettings", or
      "projects/{project}/securityHealthAnalyticsSettings"
  """
    googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule = _messages.MessageField('GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', 1)
    parent = _messages.StringField(2, required=True)