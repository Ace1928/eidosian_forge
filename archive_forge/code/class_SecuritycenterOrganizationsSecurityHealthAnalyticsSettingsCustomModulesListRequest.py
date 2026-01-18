from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModulesListRequest(_messages.Message):
    """A SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModule
  sListRequest object.

  Fields:
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last call indicating a continuation
    parent: Required. Name of parent to list custom modules. Its format is
      "organizations/{organization}/securityHealthAnalyticsSettings",
      "folders/{folder}/securityHealthAnalyticsSettings", or
      "projects/{project}/securityHealthAnalyticsSettings"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)