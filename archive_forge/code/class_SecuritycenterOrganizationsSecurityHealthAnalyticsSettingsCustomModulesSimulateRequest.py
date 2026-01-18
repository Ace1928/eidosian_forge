from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModulesSimulateRequest(_messages.Message):
    """A SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModule
  sSimulateRequest object.

  Fields:
    parent: Required. The relative resource name of the organization, project,
      or folder. For more information about relative resource names, see
      [Relative Resource Name](https://cloud.google.com/apis/design/resource_n
      ames#relative_resource_name) Example: `organizations/{organization_id}`
    simulateSecurityHealthAnalyticsCustomModuleRequest: A
      SimulateSecurityHealthAnalyticsCustomModuleRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    simulateSecurityHealthAnalyticsCustomModuleRequest = _messages.MessageField('SimulateSecurityHealthAnalyticsCustomModuleRequest', 2)