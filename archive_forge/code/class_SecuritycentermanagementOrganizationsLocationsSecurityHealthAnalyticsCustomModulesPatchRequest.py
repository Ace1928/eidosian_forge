from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesPatchRequest(_messages.Message):
    """A SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsC
  ustomModulesPatchRequest object.

  Fields:
    name: Identifier. The resource name of the custom module. Its format is "o
      rganizations/{organization}/locations/{location}/securityHealthAnalytics
      CustomModules/{security_health_analytics_custom_module}", or "folders/{f
      older}/locations/{location}/securityHealthAnalyticsCustomModules/{securi
      ty_health_analytics_custom_module}", or "projects/{project}/locations/{l
      ocation}/securityHealthAnalyticsCustomModules/{security_health_analytics
      _custom_module}" The id {customModule} is server-generated and is not
      user settable. It will be a numeric id containing 1-20 digits.
    securityHealthAnalyticsCustomModule: A SecurityHealthAnalyticsCustomModule
      resource to be passed as the request body.
    updateMask: Required. The list of fields to be updated. The only fields
      that can be updated are `enablement_state` and `custom_config`. If empty
      or set to the wildcard value `*`, both `enablement_state` and
      `custom_config` are updated.
    validateOnly: Optional. When set to true, only validations (including IAM
      checks) will done for the request (module will not be updated). An OK
      response indicates the request is valid while an error response
      indicates the request is invalid. Note that a subsequent request to
      actually update the module could still fail because 1. the state could
      have changed (e.g. IAM permission lost) or 2. A failure occurred while
      trying to update the module.
  """
    name = _messages.StringField(1, required=True)
    securityHealthAnalyticsCustomModule = _messages.MessageField('SecurityHealthAnalyticsCustomModule', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)