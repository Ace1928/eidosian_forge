from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModulesTestRequest(_messages.Message):
    """A SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCustomModule
  sTestRequest object.

  Fields:
    name: Required. Name of the custom module to test. Its format is "organiza
      tions/[organization_id]/securityHealthAnalyticsSettings/customModules/[m
      odule_id]". If the custom_module field is empty, it is assumed that the
      custom module already exists; otherwise the specified custom_module will
      be used.
    testSecurityHealthAnalyticsCustomModuleRequest: A
      TestSecurityHealthAnalyticsCustomModuleRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    testSecurityHealthAnalyticsCustomModuleRequest = _messages.MessageField('TestSecurityHealthAnalyticsCustomModuleRequest', 2)