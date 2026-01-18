from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApiConfig(_messages.Message):
    """An API Configuration is a combination of settings for both the Managed
  Service and Gateways serving this API Config.

  Enums:
    StateValueValuesEnum: Output only. State of the API Config.

  Messages:
    LabelsValue: Optional. Resource labels to represent user-provided
      metadata. Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources

  Fields:
    createTime: Output only. Created time.
    displayName: Optional. Display name.
    gatewayConfig: Immutable. Gateway specific configuration.
    gatewayServiceAccount: Immutable. The Google Cloud IAM Service Account
      that Gateways serving this config should use to authenticate to other
      services. This may either be the Service Account's email
      (`{ACCOUNT_ID}@{PROJECT}.iam.gserviceaccount.com`) or its full resource
      name (`projects/{PROJECT}/accounts/{UNIQUE_ID}`). This is most often
      used when the service is a GCP resource such as a Cloud Run Service or
      an IAP-secured service.
    grpcServices: Optional. gRPC service definition files. If specified,
      openapi_documents must not be included.
    labels: Optional. Resource labels to represent user-provided metadata.
      Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources
    managedServiceConfigs: Optional. Service Configuration files. At least one
      must be included when using gRPC service definitions. See
      https://cloud.google.com/endpoints/docs/grpc/grpc-service-
      config#service_configuration_overview for the expected file contents. If
      multiple files are specified, the files are merged with the following
      rules: * All singular scalar fields are merged using "last one wins"
      semantics in the order of the files uploaded. * Repeated fields are
      concatenated. * Singular embedded messages are merged using these rules
      for nested fields.
    name: Output only. Resource name of the API Config. Format:
      projects/{project}/locations/global/apis/{api}/configs/{api_config}
    openapiDocuments: Optional. OpenAPI specification documents. If specified,
      grpc_services and managed_service_configs must not be included.
    serviceConfigId: Output only. The ID of the associated Service Config (
      https://cloud.google.com/service-infrastructure/docs/glossary#config).
    state: Output only. State of the API Config.
    updateTime: Output only. Updated time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the API Config.

    Values:
      STATE_UNSPECIFIED: API Config does not have a state yet.
      CREATING: API Config is being created and deployed to the API
        Controller.
      ACTIVE: API Config is ready for use by Gateways.
      FAILED: API Config creation failed.
      DELETING: API Config is being deleted.
      UPDATING: API Config is being updated.
      ACTIVATING: API Config settings are being activated in downstream
        systems. API Configs in this state cannot be used by Gateways.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        FAILED = 3
        DELETING = 4
        UPDATING = 5
        ACTIVATING = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user-provided metadata. Refer
    to cloud documentation on labels for more details.
    https://cloud.google.com/compute/docs/labeling-resources

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    gatewayConfig = _messages.MessageField('ApigatewayGatewayConfig', 3)
    gatewayServiceAccount = _messages.StringField(4)
    grpcServices = _messages.MessageField('ApigatewayApiConfigGrpcServiceDefinition', 5, repeated=True)
    labels = _messages.MessageField('LabelsValue', 6)
    managedServiceConfigs = _messages.MessageField('ApigatewayApiConfigFile', 7, repeated=True)
    name = _messages.StringField(8)
    openapiDocuments = _messages.MessageField('ApigatewayApiConfigOpenApiDocument', 9, repeated=True)
    serviceConfigId = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    updateTime = _messages.StringField(12)