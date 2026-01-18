from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeployedModel(_messages.Message):
    """A deployment of a Model. Endpoints contain one or more DeployedModels.

  Fields:
    automaticResources: A description of resources that to large degree are
      decided by Vertex AI, and require only a modest additional
      configuration.
    createTime: Output only. Timestamp when the DeployedModel was created.
    dedicatedResources: A description of resources that are dedicated to the
      DeployedModel, and that need a higher degree of manual configuration.
    disableContainerLogging: For custom-trained Models and AutoML Tabular
      Models, the container of the DeployedModel instances will send `stderr`
      and `stdout` streams to Cloud Logging by default. Please note that the
      logs incur cost, which are subject to [Cloud Logging
      pricing](https://cloud.google.com/logging/pricing). User can disable
      container logging by setting this flag to true.
    disableExplanations: If true, deploy the model without explainable
      feature, regardless the existence of Model.explanation_spec or
      explanation_spec.
    displayName: The display name of the DeployedModel. If not provided upon
      creation, the Model's display_name is used.
    enableAccessLogging: If true, online prediction access logs are sent to
      Cloud Logging. These logs are like standard server access logs,
      containing information like timestamp and latency for each prediction
      request. Note that logs may incur a cost, especially if your project
      receives prediction requests at a high queries per second rate (QPS).
      Estimate your costs before enabling this option.
    explanationSpec: Explanation configuration for this DeployedModel. When
      deploying a Model using EndpointService.DeployModel, this value
      overrides the value of Model.explanation_spec. All fields of
      explanation_spec are optional in the request. If a field of
      explanation_spec is not populated, the value of the same field of
      Model.explanation_spec is inherited. If the corresponding
      Model.explanation_spec is not populated, all fields of the
      explanation_spec will be used for the explanation configuration.
    id: Immutable. The ID of the DeployedModel. If not provided upon
      deployment, Vertex AI will generate a value for this ID. This value
      should be 1-10 characters, and valid characters are `/[0-9]/`.
    model: Required. The resource name of the Model that this is the
      deployment of. Note that the Model may be in a different location than
      the DeployedModel's Endpoint. The resource name may contain version id
      or version alias to specify the version. Example:
      `projects/{project}/locations/{location}/models/{model}@2` or
      `projects/{project}/locations/{location}/models/{model}@golden` if no
      version is specified, the default version will be deployed.
    modelVersionId: Output only. The version ID of the model that is deployed.
    privateEndpoints: Output only. Provide paths for users to send
      predict/explain/health requests directly to the deployed model services
      running on Cloud via private services access. This field is populated if
      network is configured.
    serviceAccount: The service account that the DeployedModel's container
      runs as. Specify the email address of the service account. If this
      service account is not specified, the container runs as a service
      account that doesn't have access to the resource project. Users
      deploying the Model must have the `iam.serviceAccounts.actAs` permission
      on this service account.
    sharedResources: The resource name of the shared DeploymentResourcePool to
      deploy on. Format: `projects/{project}/locations/{location}/deploymentRe
      sourcePools/{deployment_resource_pool}`
  """
    automaticResources = _messages.MessageField('GoogleCloudAiplatformV1AutomaticResources', 1)
    createTime = _messages.StringField(2)
    dedicatedResources = _messages.MessageField('GoogleCloudAiplatformV1DedicatedResources', 3)
    disableContainerLogging = _messages.BooleanField(4)
    disableExplanations = _messages.BooleanField(5)
    displayName = _messages.StringField(6)
    enableAccessLogging = _messages.BooleanField(7)
    explanationSpec = _messages.MessageField('GoogleCloudAiplatformV1ExplanationSpec', 8)
    id = _messages.StringField(9)
    model = _messages.StringField(10)
    modelVersionId = _messages.StringField(11)
    privateEndpoints = _messages.MessageField('GoogleCloudAiplatformV1PrivateEndpoints', 12)
    serviceAccount = _messages.StringField(13)
    sharedResources = _messages.StringField(14)