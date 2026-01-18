from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModulesDeleteRequest(_messages.Message):
    """A SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModule
  sDeleteRequest object.

  Fields:
    name: Required. Name of the custom module to delete. Its format is "organi
      zations/{organization}/securityHealthAnalyticsSettings/customModules/{cu
      stomModule}", "folders/{folder}/securityHealthAnalyticsSettings/customMo
      dules/{customModule}", or "projects/{project}/securityHealthAnalyticsSet
      tings/customModules/{customModule}"
  """
    name = _messages.StringField(1, required=True)