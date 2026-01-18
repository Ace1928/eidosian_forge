from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Version(_messages.Message):
    """Represents a version of the model. Each version is a trained model
  deployed in the cloud, ready to handle prediction requests. A model can have
  multiple versions. You can get information about all of the versions of a
  given model by calling projects.models.versions.list.

  Enums:
    FrameworkValueValuesEnum: Optional. The machine learning framework AI
      Platform uses to train this version of the model. Valid values are
      `TENSORFLOW`, `SCIKIT_LEARN`, `XGBOOST`. If you do not specify a
      framework, AI Platform will analyze files in the deployment_uri to
      determine a framework. If you choose `SCIKIT_LEARN` or `XGBOOST`, you
      must also set the runtime version of the model to 1.4 or greater. Do
      **not** specify a framework if you're deploying a [custom prediction
      routine](/ai-platform/prediction/docs/custom-prediction-routines) or if
      you're using a [custom container](/ai-platform/prediction/docs/use-
      custom-container).
    StateValueValuesEnum: Output only. The state of a version.

  Messages:
    LabelsValue: Optional. One or more labels that you can add, to organize
      your model versions. Each label is a key-value pair, where both the key
      and the value are arbitrary strings that you supply. For more
      information, see the documentation on using labels. Note that this field
      is not updatable for mls1* models.

  Fields:
    acceleratorConfig: Optional. Accelerator config for using GPUs for online
      prediction (beta). Only specify this field if you have specified a
      Compute Engine (N1) machine type in the `machineType` field. Learn more
      about [using GPUs for online prediction](/ml-engine/docs/machine-types-
      online-prediction#gpus).
    autoScaling: Automatically scale the number of nodes used to serve the
      model in response to increases and decreases in traffic. Care should be
      taken to ramp up traffic according to the model's ability to scale or
      you will start seeing increases in latency and 429 response codes.
    container: Optional. Specifies a custom container to use for serving
      predictions. If you specify this field, then `machineType` is required.
      If you specify this field, then `deploymentUri` is optional. If you
      specify this field, then you must not specify `runtimeVersion`,
      `packageUris`, `framework`, `pythonVersion`, or `predictionClass`.
    createTime: Output only. The time the version was created.
    deploymentUri: The Cloud Storage URI of a directory containing trained
      model artifacts to be used to create the model version. See the [guide
      to deploying models](/ai-platform/prediction/docs/deploying-models) for
      more information. The total number of files under this directory must
      not exceed 1000. During projects.models.versions.create, AI Platform
      Prediction copies all files from the specified directory to a location
      managed by the service. From then on, AI Platform Prediction uses these
      copies of the model artifacts to serve predictions, not the original
      files in Cloud Storage, so this location is useful only as a historical
      record. If you specify container, then this field is optional.
      Otherwise, it is required. Learn [how to use this field with a custom
      container](/ai-platform/prediction/docs/custom-container-
      requirements#artifacts).
    description: Optional. The description specified for the version when it
      was created.
    errorMessage: Output only. The details of a failure or a cancellation.
    etag: `etag` is used for optimistic concurrency control as a way to help
      prevent simultaneous updates of a model from overwriting each other. It
      is strongly suggested that systems make use of the `etag` in the read-
      modify-write cycle to perform model updates in order to avoid race
      conditions: An `etag` is returned in the response to `GetVersion`, and
      systems are expected to put that etag in the request to `UpdateVersion`
      to ensure that their change will be applied to the model as intended.
    explanationConfig: Optional. Configures explainability features on the
      model's version. Some explanation features require additional metadata
      to be loaded as part of the model payload.
    framework: Optional. The machine learning framework AI Platform uses to
      train this version of the model. Valid values are `TENSORFLOW`,
      `SCIKIT_LEARN`, `XGBOOST`. If you do not specify a framework, AI
      Platform will analyze files in the deployment_uri to determine a
      framework. If you choose `SCIKIT_LEARN` or `XGBOOST`, you must also set
      the runtime version of the model to 1.4 or greater. Do **not** specify a
      framework if you're deploying a [custom prediction routine](/ai-
      platform/prediction/docs/custom-prediction-routines) or if you're using
      a [custom container](/ai-platform/prediction/docs/use-custom-container).
    imageUri: Optional. The docker image to run for custom serving container.
      This image must be in Google Container Registry.
    isDefault: Output only. If true, this version will be used to handle
      prediction requests that do not specify a version. You can change the
      default version by calling projects.methods.versions.setDefault.
    labels: Optional. One or more labels that you can add, to organize your
      model versions. Each label is a key-value pair, where both the key and
      the value are arbitrary strings that you supply. For more information,
      see the documentation on using labels. Note that this field is not
      updatable for mls1* models.
    lastMigrationModelId: Output only. The [AI Platform (Unified)
      `Model`](https://cloud.google.com/ai-platform-
      unified/docs/reference/rest/v1beta1/projects.locations.models) ID for
      the last [model migration](https://cloud.google.com/ai-platform-
      unified/docs/start/migrating-to-ai-platform-unified).
    lastMigrationTime: Output only. The last time this version was
      successfully [migrated to AI Platform
      (Unified)](https://cloud.google.com/ai-platform-
      unified/docs/start/migrating-to-ai-platform-unified).
    lastUseTime: Output only. The time the version was last used for
      prediction.
    machineType: Optional. The type of machine on which to serve the model.
      Currently only applies to online prediction service. To learn about
      valid values for this field, read [Choosing a machine type for online
      prediction](/ai-platform/prediction/docs/machine-types-online-
      prediction). If this field is not specified and you are using a
      [regional endpoint](/ai-platform/prediction/docs/regional-endpoints),
      then the machine type defaults to `n1-standard-2`. If this field is not
      specified and you are using the global endpoint (`ml.googleapis.com`),
      then the machine type defaults to `mls1-c1-m2`.
    manualScaling: Manually select the number of nodes to use for serving the
      model. You should generally use `auto_scaling` with an appropriate
      `min_nodes` instead, but this option is available if you want more
      predictable billing. Beware that latency and error rates will increase
      if the traffic exceeds that capability of the system to serve it based
      on the selected number of nodes.
    modelClass: A string attribute.
    name: Required. The name specified for the version when it was created.
      The version name must be unique within the model it is created in.
    packageUris: Optional. Cloud Storage paths (`gs://...`) of packages for
      [custom prediction routines](/ml-engine/docs/tensorflow/custom-
      prediction-routines) or [scikit-learn pipelines with custom code](/ml-
      engine/docs/scikit/exporting-for-prediction#custom-pipeline-code). For a
      custom prediction routine, one of these packages must contain your
      Predictor class (see
      [`predictionClass`](#Version.FIELDS.prediction_class)). Additionally,
      include any dependencies used by your Predictor or scikit-learn pipeline
      uses that are not already included in your selected [runtime
      version](/ml-engine/docs/tensorflow/runtime-version-list). If you
      specify this field, you must also set
      [`runtimeVersion`](#Version.FIELDS.runtime_version) to 1.4 or greater.
    predictionClass: Optional. The fully qualified name
      (module_name.class_name) of a class that implements the Predictor
      interface described in this reference field. The module containing this
      class should be included in a package provided to the [`packageUris`
      field](#Version.FIELDS.package_uris). Specify this field if and only if
      you are deploying a [custom prediction routine (beta)](/ml-
      engine/docs/tensorflow/custom-prediction-routines). If you specify this
      field, you must set [`runtimeVersion`](#Version.FIELDS.runtime_version)
      to 1.4 or greater and you must set `machineType` to a [legacy (MLS1)
      machine type](/ml-engine/docs/machine-types-online-prediction). The
      following code sample provides the Predictor interface: class
      Predictor(object): " " "Interface for constructing custom predictors." "
      " def predict(self, instances, **kwargs): " " "Performs custom
      prediction. Instances are the decoded values from the request. They have
      already been deserialized from JSON. Args: instances: A list of
      prediction input instances. **kwargs: A dictionary of keyword args
      provided as additional fields on the predict request body. Returns: A
      list of outputs containing the prediction results. This list must be
      JSON serializable. " " " raise NotImplementedError() @classmethod def
      from_path(cls, model_dir): " " "Creates an instance of Predictor using
      the given path. Loading of the predictor should be done in this method.
      Args: model_dir: The local directory that contains the exported model
      file along with any additional files uploaded when creating the version
      resource. Returns: An instance implementing this Predictor class. " " "
      raise NotImplementedError() Learn more about [the Predictor interface
      and custom prediction routines](/ml-engine/docs/tensorflow/custom-
      prediction-routines).
    pythonVersion: Required. The version of Python used in prediction. The
      following Python versions are available: * Python '3.7' is available
      when `runtime_version` is set to '1.15' or later. * Python '3.5' is
      available when `runtime_version` is set to a version from '1.4' to
      '1.14'. * Python '2.7' is available when `runtime_version` is set to
      '1.15' or earlier. Read more about the Python versions available for
      [each runtime version](/ml-engine/docs/runtime-version-list).
    requestLoggingConfig: Optional. *Only* specify this field in a
      projects.models.versions.patch request. Specifying it in a
      projects.models.versions.create request has no effect. Configures the
      request-response pair logging on predictions from this Version.
    routes: Optional. Specifies paths on a custom container's HTTP server
      where AI Platform Prediction sends certain requests. If you specify this
      field, then you must also specify the `container` field. If you specify
      the `container` field and do not specify this field, it defaults to the
      following: ```json { "predict":
      "/v1/models/MODEL/versions/VERSION:predict", "health":
      "/v1/models/MODEL/versions/VERSION" } ``` See RouteMap for more details
      about these default values.
    runtimeVersion: Required. The AI Platform runtime version to use for this
      deployment. For more information, see the [runtime version list](/ml-
      engine/docs/runtime-version-list) and [how to manage runtime
      versions](/ml-engine/docs/versioning).
    serviceAccount: Optional. Specifies the service account for resource
      access control. If you specify this field, then you must also specify
      either the `containerSpec` or the `predictionClass` field. Learn more
      about [using a custom service account](/ai-
      platform/prediction/docs/custom-service-account).
    state: Output only. The state of a version.
  """

    class FrameworkValueValuesEnum(_messages.Enum):
        """Optional. The machine learning framework AI Platform uses to train
    this version of the model. Valid values are `TENSORFLOW`, `SCIKIT_LEARN`,
    `XGBOOST`. If you do not specify a framework, AI Platform will analyze
    files in the deployment_uri to determine a framework. If you choose
    `SCIKIT_LEARN` or `XGBOOST`, you must also set the runtime version of the
    model to 1.4 or greater. Do **not** specify a framework if you're
    deploying a [custom prediction routine](/ai-
    platform/prediction/docs/custom-prediction-routines) or if you're using a
    [custom container](/ai-platform/prediction/docs/use-custom-container).

    Values:
      FRAMEWORK_UNSPECIFIED: Unspecified framework. Assigns a value based on
        the file suffix.
      TENSORFLOW: Tensorflow framework.
      SCIKIT_LEARN: Scikit-learn framework.
      XGBOOST: XGBoost framework.
    """
        FRAMEWORK_UNSPECIFIED = 0
        TENSORFLOW = 1
        SCIKIT_LEARN = 2
        XGBOOST = 3

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of a version.

    Values:
      UNKNOWN: The version state is unspecified.
      READY: The version is ready for prediction.
      CREATING: The version is being created. New UpdateVersion and
        DeleteVersion requests will fail if a version is in the CREATING
        state.
      FAILED: The version failed to be created, possibly cancelled.
        `error_message` should contain the details of the failure.
      DELETING: The version is being deleted. New UpdateVersion and
        DeleteVersion requests will fail if a version is in the DELETING
        state.
      UPDATING: The version is being updated. New UpdateVersion and
        DeleteVersion requests will fail if a version is in the UPDATING
        state.
    """
        UNKNOWN = 0
        READY = 1
        CREATING = 2
        FAILED = 3
        DELETING = 4
        UPDATING = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. One or more labels that you can add, to organize your model
    versions. Each label is a key-value pair, where both the key and the value
    are arbitrary strings that you supply. For more information, see the
    documentation on using labels. Note that this field is not updatable for
    mls1* models.

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
    acceleratorConfig = _messages.MessageField('GoogleCloudMlV1AcceleratorConfig', 1)
    autoScaling = _messages.MessageField('GoogleCloudMlV1AutoScaling', 2)
    container = _messages.MessageField('GoogleCloudMlV1ContainerSpec', 3)
    createTime = _messages.StringField(4)
    deploymentUri = _messages.StringField(5)
    description = _messages.StringField(6)
    errorMessage = _messages.StringField(7)
    etag = _messages.BytesField(8)
    explanationConfig = _messages.MessageField('GoogleCloudMlV1ExplanationConfig', 9)
    framework = _messages.EnumField('FrameworkValueValuesEnum', 10)
    imageUri = _messages.StringField(11)
    isDefault = _messages.BooleanField(12)
    labels = _messages.MessageField('LabelsValue', 13)
    lastMigrationModelId = _messages.StringField(14)
    lastMigrationTime = _messages.StringField(15)
    lastUseTime = _messages.StringField(16)
    machineType = _messages.StringField(17)
    manualScaling = _messages.MessageField('GoogleCloudMlV1ManualScaling', 18)
    modelClass = _messages.StringField(19)
    name = _messages.StringField(20)
    packageUris = _messages.StringField(21, repeated=True)
    predictionClass = _messages.StringField(22)
    pythonVersion = _messages.StringField(23)
    requestLoggingConfig = _messages.MessageField('GoogleCloudMlV1RequestLoggingConfig', 24)
    routes = _messages.MessageField('GoogleCloudMlV1RouteMap', 25)
    runtimeVersion = _messages.StringField(26)
    serviceAccount = _messages.StringField(27)
    state = _messages.EnumField('StateValueValuesEnum', 28)