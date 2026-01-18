from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildConfig(_messages.Message):
    """Describes the Build step of the function that builds a container from
  the given source.

  Enums:
    DockerRegistryValueValuesEnum: Docker Registry to use for this deployment.
      This configuration is only applicable to 1st Gen functions, 2nd Gen
      functions can only use Artifact Registry. If unspecified, it defaults to
      `ARTIFACT_REGISTRY`. If `docker_repository` field is specified, this
      field should either be left unspecified or set to `ARTIFACT_REGISTRY`.

  Messages:
    EnvironmentVariablesValue: User-provided build-time environment variables
      for the function

  Fields:
    automaticUpdatePolicy: A AutomaticUpdatePolicy attribute.
    build: Output only. The Cloud Build name of the latest successful
      deployment of the function.
    dockerRegistry: Docker Registry to use for this deployment. This
      configuration is only applicable to 1st Gen functions, 2nd Gen functions
      can only use Artifact Registry. If unspecified, it defaults to
      `ARTIFACT_REGISTRY`. If `docker_repository` field is specified, this
      field should either be left unspecified or set to `ARTIFACT_REGISTRY`.
    dockerRepository: Repository in Artifact Registry to which the function
      docker image will be pushed after it is built by Cloud Build. If
      specified by user, it is created and managed by user with a customer
      managed encryption key. Otherwise, GCF will create and use a repository
      named 'gcf-artifacts' for every deployed region. It must match the
      pattern
      `projects/{project}/locations/{location}/repositories/{repository}`.
      Cross-project repositories are not supported. Cross-location
      repositories are not supported. Repository format must be 'DOCKER'.
    entryPoint: The name of the function (as defined in source code) that will
      be executed. Defaults to the resource name suffix, if not specified. For
      backward compatibility, if function with given name is not found, then
      the system will try to use function named "function". For Node.js this
      is name of a function exported by the module specified in
      `source_location`.
    environmentVariables: User-provided build-time environment variables for
      the function
    onDeployUpdatePolicy: A OnDeployUpdatePolicy attribute.
    runtime: The runtime in which to run the function. Required when deploying
      a new function, optional when updating an existing function. For a
      complete list of possible choices, see the [`gcloud` command reference](
      https://cloud.google.com/sdk/gcloud/reference/functions/deploy#--
      runtime).
    serviceAccount: [Preview] Service account to be used for building the
      container
    source: The location of the function source code.
    sourceProvenance: Output only. A permanent fixed identifier for source.
    sourceToken: An identifier for Firebase function sources. Disclaimer: This
      field is only supported for Firebase function deployments.
    workerPool: Name of the Cloud Build Custom Worker Pool that should be used
      to build the function. The format of this field is
      `projects/{project}/locations/{region}/workerPools/{workerPool}` where
      {project} and {region} are the project id and region respectively where
      the worker pool is defined and {workerPool} is the short name of the
      worker pool. If the project id is not the same as the function, then the
      Cloud Functions Service Agent (service-@gcf-admin-
      robot.iam.gserviceaccount.com) must be granted the role Cloud Build
      Custom Workers Builder (roles/cloudbuild.customworkers.builder) in the
      project.
  """

    class DockerRegistryValueValuesEnum(_messages.Enum):
        """Docker Registry to use for this deployment. This configuration is only
    applicable to 1st Gen functions, 2nd Gen functions can only use Artifact
    Registry. If unspecified, it defaults to `ARTIFACT_REGISTRY`. If
    `docker_repository` field is specified, this field should either be left
    unspecified or set to `ARTIFACT_REGISTRY`.

    Values:
      DOCKER_REGISTRY_UNSPECIFIED: Unspecified.
      CONTAINER_REGISTRY: Docker images will be stored in multi-regional
        Container Registry repositories named `gcf`.
      ARTIFACT_REGISTRY: Docker images will be stored in regional Artifact
        Registry repositories. By default, GCF will create and use
        repositories named `gcf-artifacts` in every region in which a function
        is deployed. But the repository to use can also be specified by the
        user using the `docker_repository` field.
    """
        DOCKER_REGISTRY_UNSPECIFIED = 0
        CONTAINER_REGISTRY = 1
        ARTIFACT_REGISTRY = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentVariablesValue(_messages.Message):
        """User-provided build-time environment variables for the function

    Messages:
      AdditionalProperty: An additional property for a
        EnvironmentVariablesValue object.

    Fields:
      additionalProperties: Additional properties of type
        EnvironmentVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvironmentVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    automaticUpdatePolicy = _messages.MessageField('AutomaticUpdatePolicy', 1)
    build = _messages.StringField(2)
    dockerRegistry = _messages.EnumField('DockerRegistryValueValuesEnum', 3)
    dockerRepository = _messages.StringField(4)
    entryPoint = _messages.StringField(5)
    environmentVariables = _messages.MessageField('EnvironmentVariablesValue', 6)
    onDeployUpdatePolicy = _messages.MessageField('OnDeployUpdatePolicy', 7)
    runtime = _messages.StringField(8)
    serviceAccount = _messages.StringField(9)
    source = _messages.MessageField('Source', 10)
    sourceProvenance = _messages.MessageField('SourceProvenance', 11)
    sourceToken = _messages.StringField(12)
    workerPool = _messages.StringField(13)