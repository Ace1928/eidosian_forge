from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchPredictionJob(_messages.Message):
    """A job that uses a Model to produce predictions on multiple input
  instances. If predictions for significant portion of the instances fail, the
  job may finish without attempting predictions for all remaining instances.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the job.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize
      BatchPredictionJobs. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    completionStats: Output only. Statistics on completed and failed
      prediction instances.
    createTime: Output only. Time when the BatchPredictionJob was created.
    dedicatedResources: The config of resources used by the Model during the
      batch prediction. If the Model supports DEDICATED_RESOURCES this config
      may be provided (and the job will use these resources), if the Model
      doesn't support AUTOMATIC_RESOURCES, this config must be provided.
    disableContainerLogging: For custom-trained Models and AutoML Tabular
      Models, the container of the DeployedModel instances will send `stderr`
      and `stdout` streams to Cloud Logging by default. Please note that the
      logs incur cost, which are subject to [Cloud Logging
      pricing](https://cloud.google.com/logging/pricing). User can disable
      container logging by setting this flag to true.
    displayName: Required. The user-defined name of this BatchPredictionJob.
    encryptionSpec: Customer-managed encryption key options for a
      BatchPredictionJob. If this is set, then all resources created by the
      BatchPredictionJob will be encrypted with the provided encryption key.
    endTime: Output only. Time when the BatchPredictionJob entered any of the
      following states: `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
      `JOB_STATE_CANCELLED`.
    error: Output only. Only populated when the job's state is
      JOB_STATE_FAILED or JOB_STATE_CANCELLED.
    explanationSpec: Explanation configuration for this BatchPredictionJob.
      Can be specified only if generate_explanation is set to `true`. This
      value overrides the value of Model.explanation_spec. All fields of
      explanation_spec are optional in the request. If a field of the
      explanation_spec object is not populated, the corresponding field of the
      Model.explanation_spec object is inherited.
    generateExplanation: Generate explanation with the batch prediction
      results. When set to `true`, the batch prediction output changes based
      on the `predictions_format` field of the
      BatchPredictionJob.output_config object: * `bigquery`: output includes a
      column named `explanation`. The value is a struct that conforms to the
      Explanation object. * `jsonl`: The JSON objects on each line include an
      additional entry keyed `explanation`. The value of the entry is a JSON
      object that conforms to the Explanation object. * `csv`: Generating
      explanations for CSV format is not supported. If this field is set to
      true, either the Model.explanation_spec or explanation_spec must be
      populated.
    inputConfig: Required. Input configuration of the instances on which
      predictions are performed. The schema of any single instance may be
      specified via the Model's PredictSchemata's instance_schema_uri.
    instanceConfig: Configuration for how to convert batch prediction input
      instances to the prediction instances that are sent to the Model.
    labels: The labels with user-defined metadata to organize
      BatchPredictionJobs. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    manualBatchTuningParameters: Immutable. Parameters configuring the batch
      behavior. Currently only applicable when dedicated_resources are used
      (in other cases Vertex AI does the tuning itself).
    model: The name of the Model resource that produces the predictions via
      this job, must share the same ancestor Location. Starting this job has
      no impact on any existing deployments of the Model and their resources.
      Exactly one of model and unmanaged_container_model must be set. The
      model resource name may contain version id or version alias to specify
      the version. Example:
      `projects/{project}/locations/{location}/models/{model}@2` or
      `projects/{project}/locations/{location}/models/{model}@golden` if no
      version is specified, the default version will be deployed. The model
      resource could also be a publisher model. Example:
      `publishers/{publisher}/models/{model}` or `projects/{project}/locations
      /{location}/publishers/{publisher}/models/{model}`
    modelMonitoringConfig: Model monitoring config will be used for analysis
      model behaviors, based on the input and output to the batch prediction
      job, as well as the provided training dataset.
    modelMonitoringStatsAnomalies: Get batch prediction job monitoring
      statistics.
    modelMonitoringStatus: Output only. The running status of the model
      monitoring pipeline.
    modelParameters: The parameters that govern the predictions. The schema of
      the parameters may be specified via the Model's PredictSchemata's
      parameters_schema_uri.
    modelVersionId: Output only. The version ID of the Model that produces the
      predictions via this job.
    name: Output only. Resource name of the BatchPredictionJob.
    outputConfig: Required. The Configuration specifying where output
      predictions should be written. The schema of any single prediction may
      be specified as a concatenation of Model's PredictSchemata's
      instance_schema_uri and prediction_schema_uri.
    outputInfo: Output only. Information further describing the output of this
      job.
    partialFailures: Output only. Partial failures encountered. For example,
      single files that can't be read. This field never exceeds 20 entries.
      Status details fields contain standard Google Cloud error details.
    resourcesConsumed: Output only. Information about resources that had been
      consumed by this job. Provided in real time at best effort basis, as
      well as a final value once the job completes. Note: This field currently
      may be not populated for batch predictions that use AutoML Models.
    serviceAccount: The service account that the DeployedModel's container
      runs as. If not specified, a system generated one will be used, which
      has minimal permissions and the custom container, if used, may not have
      enough permission to access other Google Cloud resources. Users
      deploying the Model must have the `iam.serviceAccounts.actAs` permission
      on this service account.
    startTime: Output only. Time when the BatchPredictionJob for the first
      time entered the `JOB_STATE_RUNNING` state.
    state: Output only. The detailed state of the job.
    unmanagedContainerModel: Contains model information necessary to perform
      batch prediction without requiring uploading to model registry. Exactly
      one of model and unmanaged_container_model must be set.
    updateTime: Output only. Time when the BatchPredictionJob was most
      recently updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the job.

    Values:
      JOB_STATE_UNSPECIFIED: The job state is unspecified.
      JOB_STATE_QUEUED: The job has been just created or resumed and
        processing has not yet begun.
      JOB_STATE_PENDING: The service is preparing to run the job.
      JOB_STATE_RUNNING: The job is in progress.
      JOB_STATE_SUCCEEDED: The job completed successfully.
      JOB_STATE_FAILED: The job failed.
      JOB_STATE_CANCELLING: The job is being cancelled. From this state the
        job may only go to either `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED` or
        `JOB_STATE_CANCELLED`.
      JOB_STATE_CANCELLED: The job has been cancelled.
      JOB_STATE_PAUSED: The job has been stopped, and can be resumed.
      JOB_STATE_EXPIRED: The job has expired.
      JOB_STATE_UPDATING: The job is being updated. Only jobs in the `RUNNING`
        state can be updated. After updating, the job goes back to the
        `RUNNING` state.
      JOB_STATE_PARTIALLY_SUCCEEDED: The job is partially succeeded, some
        results may be missing due to errors.
    """
        JOB_STATE_UNSPECIFIED = 0
        JOB_STATE_QUEUED = 1
        JOB_STATE_PENDING = 2
        JOB_STATE_RUNNING = 3
        JOB_STATE_SUCCEEDED = 4
        JOB_STATE_FAILED = 5
        JOB_STATE_CANCELLING = 6
        JOB_STATE_CANCELLED = 7
        JOB_STATE_PAUSED = 8
        JOB_STATE_EXPIRED = 9
        JOB_STATE_UPDATING = 10
        JOB_STATE_PARTIALLY_SUCCEEDED = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize BatchPredictionJobs.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. See
    https://goo.gl/xmQnxf for more information and examples of labels.

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
    completionStats = _messages.MessageField('GoogleCloudAiplatformV1beta1CompletionStats', 1)
    createTime = _messages.StringField(2)
    dedicatedResources = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchDedicatedResources', 3)
    disableContainerLogging = _messages.BooleanField(4)
    displayName = _messages.StringField(5)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 6)
    endTime = _messages.StringField(7)
    error = _messages.MessageField('GoogleRpcStatus', 8)
    explanationSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationSpec', 9)
    generateExplanation = _messages.BooleanField(10)
    inputConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchPredictionJobInputConfig', 11)
    instanceConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchPredictionJobInstanceConfig', 12)
    labels = _messages.MessageField('LabelsValue', 13)
    manualBatchTuningParameters = _messages.MessageField('GoogleCloudAiplatformV1beta1ManualBatchTuningParameters', 14)
    model = _messages.StringField(15)
    modelMonitoringConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringConfig', 16)
    modelMonitoringStatsAnomalies = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringStatsAnomalies', 17, repeated=True)
    modelMonitoringStatus = _messages.MessageField('GoogleRpcStatus', 18)
    modelParameters = _messages.MessageField('extra_types.JsonValue', 19)
    modelVersionId = _messages.StringField(20)
    name = _messages.StringField(21)
    outputConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchPredictionJobOutputConfig', 22)
    outputInfo = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchPredictionJobOutputInfo', 23)
    partialFailures = _messages.MessageField('GoogleRpcStatus', 24, repeated=True)
    resourcesConsumed = _messages.MessageField('GoogleCloudAiplatformV1beta1ResourcesConsumed', 25)
    serviceAccount = _messages.StringField(26)
    startTime = _messages.StringField(27)
    state = _messages.EnumField('StateValueValuesEnum', 28)
    unmanagedContainerModel = _messages.MessageField('GoogleCloudAiplatformV1beta1UnmanagedContainerModel', 29)
    updateTime = _messages.StringField(30)