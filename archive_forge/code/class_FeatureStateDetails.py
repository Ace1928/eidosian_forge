from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureStateDetails(_messages.Message):
    """FeatureStateDetails is a semi-structured status message for a
  declarative resource in the API.

  Enums:
    CodeValueValuesEnum: The code describes, at a high level, if the Feature
      is operating correctly. Non-`OK` codes should have details in the
      `description` describing what actions (if any) need to be taken to
      return the Feature to `OK`.

  Fields:
    anthosobservabilityFeatureState: State for the Anthos Observability
      Feature
    appdevexperienceFeatureState: State for the AppDevExperience Feature.
    authorizerFeatureState: State for the Authorizer Feature.
    cloudauditloggingFeatureState: The state of the Anthos Cloud Audit Logging
      feature.
    code: The code describes, at a high level, if the Feature is operating
      correctly. Non-`OK` codes should have details in the `description`
      describing what actions (if any) need to be taken to return the Feature
      to `OK`.
    configmanagementFeatureState: State for the Config Management Feature.
    dataplanev2FeatureState: State for multi-cluster dataplane-v2 feature.
    description: Human readable description of the issue.
    fleetobservabilityFeatureState: State for the FleetObservability Feature.
    helloworldFeatureState: State for the Hello World Feature.
    identityserviceFeatureState: State for the AIS Feature.
    meteringFeatureState: State for the Metering Feature.
    multiclusteringressFeatureState: State for the Ingress for Anthos Feature.
    multiclusterservicediscoveryFeatureState: State for the Multi-cluster
      Service Discovery Feature.
    namespaceactuationFeatureState: State for Fleet Namespace Actuation.
    policycontrollerFeatureState: State for the Policy Controller Feature.
    rbacrolebindingactuationFeatureState: State for RBAC Role Binding
      Actuation.
    servicedirectoryFeatureState: State for the Service Directory Feature.
    servicemeshFeatureState: State for the Service Mesh Feature.
    updateTime: The last update time of this status by the controllers
    workloadcertificateFeatureState: State for the Workload Certificate
      Feature
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The code describes, at a high level, if the Feature is operating
    correctly. Non-`OK` codes should have details in the `description`
    describing what actions (if any) need to be taken to return the Feature to
    `OK`.

    Values:
      CODE_UNSPECIFIED: Not set.
      OK: No error.
      FAILED: The Feature has encountered an issue that blocks all, or a
        significant portion, of its normal operation. See the `description`
        for more details.
      WARNING: The Feature is in a state, or has encountered an issue, that
        impacts its normal operation. This state may or may not require
        intervention to resolve, see the `description` for more details.
    """
        CODE_UNSPECIFIED = 0
        OK = 1
        FAILED = 2
        WARNING = 3
    anthosobservabilityFeatureState = _messages.MessageField('AnthosObservabilityFeatureState', 1)
    appdevexperienceFeatureState = _messages.MessageField('AppDevExperienceFeatureState', 2)
    authorizerFeatureState = _messages.MessageField('AuthorizerFeatureState', 3)
    cloudauditloggingFeatureState = _messages.MessageField('CloudAuditLoggingFeatureState', 4)
    code = _messages.EnumField('CodeValueValuesEnum', 5)
    configmanagementFeatureState = _messages.MessageField('ConfigManagementFeatureState', 6)
    dataplanev2FeatureState = _messages.MessageField('DataplaneV2FeatureState', 7)
    description = _messages.StringField(8)
    fleetobservabilityFeatureState = _messages.MessageField('FleetObservabilityFeatureState', 9)
    helloworldFeatureState = _messages.MessageField('HelloWorldFeatureState', 10)
    identityserviceFeatureState = _messages.MessageField('IdentityServiceFeatureState', 11)
    meteringFeatureState = _messages.MessageField('MeteringFeatureState', 12)
    multiclusteringressFeatureState = _messages.MessageField('MultiClusterIngressFeatureState', 13)
    multiclusterservicediscoveryFeatureState = _messages.MessageField('MultiClusterServiceDiscoveryFeatureState', 14)
    namespaceactuationFeatureState = _messages.MessageField('NamespaceActuationFeatureState', 15)
    policycontrollerFeatureState = _messages.MessageField('PolicyControllerFeatureState', 16)
    rbacrolebindingactuationFeatureState = _messages.MessageField('RBACRoleBindingActuationFeatureState', 17)
    servicedirectoryFeatureState = _messages.MessageField('ServiceDirectoryFeatureState', 18)
    servicemeshFeatureState = _messages.MessageField('ServiceMeshFeatureState', 19)
    updateTime = _messages.StringField(20)
    workloadcertificateFeatureState = _messages.MessageField('WorkloadCertificateFeatureState', 21)