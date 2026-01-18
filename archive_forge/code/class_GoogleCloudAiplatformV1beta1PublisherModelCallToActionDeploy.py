from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToActionDeploy(_messages.Message):
    """Model metadata that is needed for UploadModel or
  DeployModel/CreateEndpoint requests.

  Fields:
    artifactUri: Optional. The path to the directory containing the Model
      artifact and any of its supporting files.
    automaticResources: A description of resources that to large degree are
      decided by Vertex AI, and require only a modest additional
      configuration.
    containerSpec: Optional. The specification of the container that is to be
      used when deploying this Model in Vertex AI. Not present for Large
      Models.
    dedicatedResources: A description of resources that are dedicated to the
      DeployedModel, and that need a higher degree of manual configuration.
    largeModelReference: Optional. Large model reference. When this is set,
      model_artifact_spec is not needed.
    modelDisplayName: Optional. Default model display name.
    publicArtifactUri: Optional. The signed URI for ephemeral Cloud Storage
      access to model artifact.
    sharedResources: The resource name of the shared DeploymentResourcePool to
      deploy on. Format: `projects/{project}/locations/{location}/deploymentRe
      sourcePools/{deployment_resource_pool}`
    title: Required. The title of the regional resource reference.
  """
    artifactUri = _messages.StringField(1)
    automaticResources = _messages.MessageField('GoogleCloudAiplatformV1beta1AutomaticResources', 2)
    containerSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelContainerSpec', 3)
    dedicatedResources = _messages.MessageField('GoogleCloudAiplatformV1beta1DedicatedResources', 4)
    largeModelReference = _messages.MessageField('GoogleCloudAiplatformV1beta1LargeModelReference', 5)
    modelDisplayName = _messages.StringField(6)
    publicArtifactUri = _messages.StringField(7)
    sharedResources = _messages.StringField(8)
    title = _messages.StringField(9)