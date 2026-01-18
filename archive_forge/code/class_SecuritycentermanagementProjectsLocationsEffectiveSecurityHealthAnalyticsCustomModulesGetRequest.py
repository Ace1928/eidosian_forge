from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementProjectsLocationsEffectiveSecurityHealthAnalyticsCustomModulesGetRequest(_messages.Message):
    """A SecuritycentermanagementProjectsLocationsEffectiveSecurityHealthAnalyt
  icsCustomModulesGetRequest object.

  Fields:
    name: Required. The resource name of the SHA custom module. Its format is:
      * "organizations/{organization}/locations/{location}/effectiveSecurityHe
      althAnalyticsCustomModules/{module_id}". * "folders/{folder}/locations/{
      location}/effectiveSecurityHealthAnalyticsCustomModules/{module_id}". *
      "projects/{project}/locations/{location}/effectiveSecurityHealthAnalytic
      sCustomModules/{module_id}".
  """
    name = _messages.StringField(1, required=True)