from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Model(_messages.Message):
    """A trained machine learning Model.

  Enums:
    SupportedDeploymentResourcesTypesValueListEntryValuesEnum:

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Models. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    artifactUri: Immutable. The path to the directory containing the Model
      artifact and any of its supporting files. Not required for AutoML
      Models.
    baseModelSource: Optional. User input field to specify the base model
      source. Currently it only supports specifing the Model Garden models and
      Genie models.
    containerSpec: Input only. The specification of the container that is to
      be used when deploying this Model. The specification is ingested upon
      ModelService.UploadModel, and all binaries it contains are copied and
      stored internally by Vertex AI. Not required for AutoML Models.
    createTime: Output only. Timestamp when this Model was uploaded into
      Vertex AI.
    deployedModels: Output only. The pointers to DeployedModels created from
      this Model. Note that Model could have been deployed to Endpoints in
      different Locations.
    description: The description of the Model.
    displayName: Required. The display name of the Model. The name can be up
      to 128 characters long and can consist of any UTF-8 characters.
    encryptionSpec: Customer-managed encryption key spec for a Model. If set,
      this Model and all sub-resources of this Model will be secured by this
      key.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    explanationSpec: The default explanation specification for this Model. The
      Model can be used for requesting explanation after being deployed if it
      is populated. The Model can be used for batch explanation if it is
      populated. All fields of the explanation_spec can be overridden by
      explanation_spec of DeployModelRequest.deployed_model, or
      explanation_spec of BatchPredictionJob. If the default explanation
      specification is not set for this Model, this Model can still be used
      for requesting explanation by setting explanation_spec of
      DeployModelRequest.deployed_model and for batch explanation by setting
      explanation_spec of BatchPredictionJob.
    labels: The labels with user-defined metadata to organize your Models.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    metadata: Immutable. An additional information about the Model; the schema
      of the metadata can be found in metadata_schema. Unset if the Model does
      not have any additional information.
    metadataArtifact: Output only. The resource name of the Artifact that was
      created in MetadataStore when creating the Model. The Artifact resource
      name pattern is `projects/{project}/locations/{location}/metadataStores/
      {metadata_store}/artifacts/{artifact}`.
    metadataSchemaUri: Immutable. Points to a YAML file stored on Google Cloud
      Storage describing additional information about the Model, that is
      specific to it. Unset if the Model does not have any additional
      information. The schema is defined as an OpenAPI 3.0.2 [Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). AutoML Models
      always have this field populated by Vertex AI, if no additional metadata
      is needed, this field is set to an empty string. Note: The URI given on
      output will be immutable and probably different, including the URI
      scheme, than the one given on input. The output URI will point to a
      location where the user only has a read access.
    modelSourceInfo: Output only. Source of a model. It can either be automl
      training pipeline, custom training pipeline, BigQuery ML, or saved and
      tuned from Genie or Model Garden.
    name: The resource name of the Model.
    originalModelInfo: Output only. If this Model is a copy of another Model,
      this contains info about the original.
    pipelineJob: Optional. This field is populated if the model is produced by
      a pipeline job.
    predictSchemata: The schemata that describe formats of the Model's
      predictions and explanations as given and returned via
      PredictionService.Predict and PredictionService.Explain.
    supportedDeploymentResourcesTypes: Output only. When this Model is
      deployed, its prediction resources are described by the
      `prediction_resources` field of the Endpoint.deployed_models object.
      Because not all Models support all resource configuration types, the
      configuration types this Model supports are listed here. If no
      configuration types are listed, the Model cannot be deployed to an
      Endpoint and does not support online predictions
      (PredictionService.Predict or PredictionService.Explain). Such a Model
      can serve predictions by using a BatchPredictionJob, if it has at least
      one entry each in supported_input_storage_formats and
      supported_output_storage_formats.
    supportedExportFormats: Output only. The formats in which this Model may
      be exported. If empty, this Model is not available for export.
    supportedInputStorageFormats: Output only. The formats this Model supports
      in BatchPredictionJob.input_config. If
      PredictSchemata.instance_schema_uri exists, the instances should be
      given as per that schema. The possible formats are: * `jsonl` The JSON
      Lines format, where each instance is a single line. Uses GcsSource. *
      `csv` The CSV format, where each instance is a single comma-separated
      line. The first line in the file is the header, containing comma-
      separated field names. Uses GcsSource. * `tf-record` The TFRecord
      format, where each instance is a single record in tfrecord syntax. Uses
      GcsSource. * `tf-record-gzip` Similar to `tf-record`, but the file is
      gzipped. Uses GcsSource. * `bigquery` Each instance is a single row in
      BigQuery. Uses BigQuerySource. * `file-list` Each line of the file is
      the location of an instance to process, uses `gcs_source` field of the
      InputConfig object. If this Model doesn't support any of these formats
      it means it cannot be used with a BatchPredictionJob. However, if it has
      supported_deployment_resources_types, it could serve online predictions
      by using PredictionService.Predict or PredictionService.Explain.
    supportedOutputStorageFormats: Output only. The formats this Model
      supports in BatchPredictionJob.output_config. If both
      PredictSchemata.instance_schema_uri and
      PredictSchemata.prediction_schema_uri exist, the predictions are
      returned together with their instances. In other words, the prediction
      has the original instance data first, followed by the actual prediction
      content (as per the schema). The possible formats are: * `jsonl` The
      JSON Lines format, where each prediction is a single line. Uses
      GcsDestination. * `csv` The CSV format, where each prediction is a
      single comma-separated line. The first line in the file is the header,
      containing comma-separated field names. Uses GcsDestination. *
      `bigquery` Each prediction is a single row in a BigQuery table, uses
      BigQueryDestination . If this Model doesn't support any of these formats
      it means it cannot be used with a BatchPredictionJob. However, if it has
      supported_deployment_resources_types, it could serve online predictions
      by using PredictionService.Predict or PredictionService.Explain.
    trainingPipeline: Output only. The resource name of the TrainingPipeline
      that uploaded this Model, if any.
    updateTime: Output only. Timestamp when this Model was most recently
      updated.
    versionAliases: User provided version aliases so that a model version can
      be referenced via alias (i.e. `projects/{project}/locations/{location}/m
      odels/{model_id}@{version_alias}` instead of auto-generated version id
      (i.e. `projects/{project}/locations/{location}/models/{model_id}@{versio
      n_id})`. The format is a-z{0,126}[a-z0-9] to distinguish from
      version_id. A default version alias will be created for the first
      version of the model, and there must be exactly one default version
      alias for a model.
    versionCreateTime: Output only. Timestamp when this version was created.
    versionDescription: The description of this version.
    versionId: Output only. Immutable. The version ID of the model. A new
      version is committed when a new model version is uploaded or trained
      under an existing model id. It is an auto-incrementing decimal number in
      string representation.
    versionUpdateTime: Output only. Timestamp when this version was most
      recently updated.
  """

    class SupportedDeploymentResourcesTypesValueListEntryValuesEnum(_messages.Enum):
        """SupportedDeploymentResourcesTypesValueListEntryValuesEnum enum type.

    Values:
      DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED: Should not be used.
      DEDICATED_RESOURCES: Resources that are dedicated to the DeployedModel,
        and that need a higher degree of manual configuration.
      AUTOMATIC_RESOURCES: Resources that to large degree are decided by
        Vertex AI, and require only a modest additional configuration.
      SHARED_RESOURCES: Resources that can be shared by multiple
        DeployedModels. A pre-configured DeploymentResourcePool is required.
    """
        DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED = 0
        DEDICATED_RESOURCES = 1
        AUTOMATIC_RESOURCES = 2
        SHARED_RESOURCES = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Models. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. See https://goo.gl/xmQnxf
    for more information and examples of labels.

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
    artifactUri = _messages.StringField(1)
    baseModelSource = _messages.MessageField('GoogleCloudAiplatformV1ModelBaseModelSource', 2)
    containerSpec = _messages.MessageField('GoogleCloudAiplatformV1ModelContainerSpec', 3)
    createTime = _messages.StringField(4)
    deployedModels = _messages.MessageField('GoogleCloudAiplatformV1DeployedModelRef', 5, repeated=True)
    description = _messages.StringField(6)
    displayName = _messages.StringField(7)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 8)
    etag = _messages.StringField(9)
    explanationSpec = _messages.MessageField('GoogleCloudAiplatformV1ExplanationSpec', 10)
    labels = _messages.MessageField('LabelsValue', 11)
    metadata = _messages.MessageField('extra_types.JsonValue', 12)
    metadataArtifact = _messages.StringField(13)
    metadataSchemaUri = _messages.StringField(14)
    modelSourceInfo = _messages.MessageField('GoogleCloudAiplatformV1ModelSourceInfo', 15)
    name = _messages.StringField(16)
    originalModelInfo = _messages.MessageField('GoogleCloudAiplatformV1ModelOriginalModelInfo', 17)
    pipelineJob = _messages.StringField(18)
    predictSchemata = _messages.MessageField('GoogleCloudAiplatformV1PredictSchemata', 19)
    supportedDeploymentResourcesTypes = _messages.EnumField('SupportedDeploymentResourcesTypesValueListEntryValuesEnum', 20, repeated=True)
    supportedExportFormats = _messages.MessageField('GoogleCloudAiplatformV1ModelExportFormat', 21, repeated=True)
    supportedInputStorageFormats = _messages.StringField(22, repeated=True)
    supportedOutputStorageFormats = _messages.StringField(23, repeated=True)
    trainingPipeline = _messages.StringField(24)
    updateTime = _messages.StringField(25)
    versionAliases = _messages.StringField(26, repeated=True)
    versionCreateTime = _messages.StringField(27)
    versionDescription = _messages.StringField(28)
    versionId = _messages.StringField(29)
    versionUpdateTime = _messages.StringField(30)