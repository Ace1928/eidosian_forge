from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AddonsConfig(_messages.Message):
    """Add-on configurations for the Apigee organization.

  Fields:
    advancedApiOpsConfig: Configuration for the Advanced API Ops add-on.
    analyticsConfig: Configuration for the Analytics add-on.
    apiSecurityConfig: Configuration for the API Security add-on.
    connectorsPlatformConfig: Configuration for the Connectors Platform add-
      on.
    integrationConfig: Configuration for the Integration add-on.
    monetizationConfig: Configuration for the Monetization add-on.
  """
    advancedApiOpsConfig = _messages.MessageField('GoogleCloudApigeeV1AdvancedApiOpsConfig', 1)
    analyticsConfig = _messages.MessageField('GoogleCloudApigeeV1AnalyticsConfig', 2)
    apiSecurityConfig = _messages.MessageField('GoogleCloudApigeeV1ApiSecurityConfig', 3)
    connectorsPlatformConfig = _messages.MessageField('GoogleCloudApigeeV1ConnectorsPlatformConfig', 4)
    integrationConfig = _messages.MessageField('GoogleCloudApigeeV1IntegrationConfig', 5)
    monetizationConfig = _messages.MessageField('GoogleCloudApigeeV1MonetizationConfig', 6)