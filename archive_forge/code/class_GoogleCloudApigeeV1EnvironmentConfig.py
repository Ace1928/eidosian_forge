from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EnvironmentConfig(_messages.Message):
    """A GoogleCloudApigeeV1EnvironmentConfig object.

  Messages:
    FeatureFlagsValue: Feature flags inherited from the organization and
      environment.

  Fields:
    addonsConfig: The latest runtime configurations for add-ons.
    arcConfigLocation: The location for the config blob of API Runtime
      Control, aka Envoy Adapter, for op-based authentication as a URI, e.g. a
      Cloud Storage URI. This is only used by Envoy-based gateways.
    createTime: Time that the environment configuration was created.
    dataCollectors: List of data collectors used by the deployments in the
      environment.
    debugMask: Debug mask that applies to all deployments in the environment.
    deploymentGroups: List of deployment groups in the environment.
    deployments: List of deployments in the environment.
    envScopedRevisionId: Revision ID for environment-scoped resources (e.g.
      target servers, keystores) in this config. This ID will increment any
      time a resource not scoped to a deployment group changes.
    featureFlags: Feature flags inherited from the organization and
      environment.
    flowhooks: List of flow hooks in the environment.
    forwardProxyUri: The forward proxy's url to be used by the runtime. When
      set, runtime will send requests to the target via the given forward
      proxy. This is only used by programmable gateways.
    gatewayConfigLocation: The location for the gateway config blob as a URI,
      e.g. a Cloud Storage URI. This is only used by Envoy-based gateways.
    keystores: List of keystores in the environment.
    name: Name of the environment configuration in the following format:
      `organizations/{org}/environments/{env}/configs/{config}`
    provider: Used by the Control plane to add context information to help
      detect the source of the document during diagnostics and debugging.
    pubsubTopic: Name of the PubSub topic for the environment.
    resourceReferences: List of resource references in the environment.
    resources: List of resource versions in the environment.
    revisionId: Revision ID of the environment configuration. The higher the
      value, the more recently the configuration was deployed.
    sequenceNumber: DEPRECATED: Use revision_id.
    targets: List of target servers in the environment. Disabled target
      servers are not displayed.
    traceConfig: Trace configurations. Contains config for the environment and
      config overrides for specific API proxies.
    uid: Unique ID for the environment configuration. The ID will only change
      if the environment is deleted and recreated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FeatureFlagsValue(_messages.Message):
        """Feature flags inherited from the organization and environment.

    Messages:
      AdditionalProperty: An additional property for a FeatureFlagsValue
        object.

    Fields:
      additionalProperties: Additional properties of type FeatureFlagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FeatureFlagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    addonsConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeAddonsConfig', 1)
    arcConfigLocation = _messages.StringField(2)
    createTime = _messages.StringField(3)
    dataCollectors = _messages.MessageField('GoogleCloudApigeeV1DataCollectorConfig', 4, repeated=True)
    debugMask = _messages.MessageField('GoogleCloudApigeeV1DebugMask', 5)
    deploymentGroups = _messages.MessageField('GoogleCloudApigeeV1DeploymentGroupConfig', 6, repeated=True)
    deployments = _messages.MessageField('GoogleCloudApigeeV1DeploymentConfig', 7, repeated=True)
    envScopedRevisionId = _messages.IntegerField(8)
    featureFlags = _messages.MessageField('FeatureFlagsValue', 9)
    flowhooks = _messages.MessageField('GoogleCloudApigeeV1FlowHookConfig', 10, repeated=True)
    forwardProxyUri = _messages.StringField(11)
    gatewayConfigLocation = _messages.StringField(12)
    keystores = _messages.MessageField('GoogleCloudApigeeV1KeystoreConfig', 13, repeated=True)
    name = _messages.StringField(14)
    provider = _messages.StringField(15)
    pubsubTopic = _messages.StringField(16)
    resourceReferences = _messages.MessageField('GoogleCloudApigeeV1ReferenceConfig', 17, repeated=True)
    resources = _messages.MessageField('GoogleCloudApigeeV1ResourceConfig', 18, repeated=True)
    revisionId = _messages.IntegerField(19)
    sequenceNumber = _messages.IntegerField(20)
    targets = _messages.MessageField('GoogleCloudApigeeV1TargetServerConfig', 21, repeated=True)
    traceConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeTraceConfig', 22)
    uid = _messages.StringField(23)