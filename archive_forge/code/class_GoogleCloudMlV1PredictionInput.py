from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1PredictionInput(_messages.Message):
    """Represents input parameters for a prediction job.

  Enums:
    DataFormatValueValuesEnum: Required. The format of the input data files.
    FrameworkValueValuesEnum: Optional. The framework used to train this
      model. Only needed if model_version is a GCS path. Otherwise the
      framework specified during version creation will be used.
    OutputDataFormatValueValuesEnum: Optional. Format of the output data
      files, defaults to JSON.

  Fields:
    accelerator: Optional. The type and number of accelerators to be attached
      to each machine running the job.
    batchSize: Optional. Number of records per batch, defaults to 64. The
      service will buffer batch_size number of records in memory before
      invoking one Tensorflow prediction call internally. So take the record
      size and memory available into consideration when setting this
      parameter.
    dataFormat: Required. The format of the input data files.
    framework: Optional. The framework used to train this model. Only needed
      if model_version is a GCS path. Otherwise the framework specified during
      version creation will be used.
    initialWorkerCount: Optional. The initial number of workers to be used for
      parallel processing. Defaults to 0 if one wants the service to figure
      out the number. The actual number of workers being used may change after
      the job starts depending on the autoscaling policy.
    inputPaths: Required. The Cloud Storage location of the input data files.
      May contain wildcards.
    maxWorkerCount: Optional. The maximum number of workers to be used for
      parallel processing. Defaults to 10 if not specified.
    modelName: Use this field if you want to use the default version for the
      specified model. The string must use the following format:
      `"projects/YOUR_PROJECT/models/YOUR_MODEL"`
    outputDataFormat: Optional. Format of the output data files, defaults to
      JSON.
    outputPath: Required. The output Google Cloud Storage location.
    region: Required. The Google Compute Engine region to run the prediction
      job in. See the available regions for AI Platform services.
    runtimeVersion: Optional. The AI Platform runtime version to use for this
      batch prediction. If not set, AI Platform will pick the runtime version
      used during the CreateVersion request for this model version, or choose
      the latest stable version when model version information is not
      available such as when the model is specified by uri.
    signatureName: Optional. The name of the signature defined in the
      SavedModel to use for this job. Please refer to
      [SavedModel](https://tensorflow.github.io/serving/serving_basic.html)
      for information about how to use signatures. Defaults to [DEFAULT_SERVIN
      G_SIGNATURE_DEF_KEY](https://www.tensorflow.org/api_docs/python/tf/saved
      _model/signature_constants) , which is "serving_default".
    tagsOverride: Optional. The set of tags to select which meta graph defined
      in the SavedModel to use for this job. Please refer to
      [SavedModel](https://www.tensorflow.org/serving/serving_basic) for
      information about how to use tags. Overrides the default tags when
      predicting from a deployed model version. When predicting from a model
      directory, the tag defaults to [SERVING](https://www.tensorflow.org/api_
      docs/python/tf/saved_model/tag_constants) , which is "serve".
    uri: Use this field if you want to specify a Google Cloud Storage path for
      the model to use.
    versionName: Use this field if you want to specify a version of the model
      to use. The string is formatted the same way as `model_version`, with
      the addition of the version information:
      `"projects/YOUR_PROJECT/models/YOUR_MODEL/versions/YOUR_VERSION"`
    workerType: Optional. The type of virtual machine to use for batch
      prediction job's worker nodes. It supports all machine types available
      on GCP ( https://cloud.google.com/compute/docs/machine-types), subject
      to the availability in the specific region the job runs.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """Required. The format of the input data files.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format.
      JSON: Each line of the file is a JSON dictionary representing one
        record.
      TEXT: Deprecated. Use JSON instead.
      TF_RECORD: The source file is a TFRecord file. Currently available only
        for input data.
      TF_RECORD_GZIP: The source file is a GZIP-compressed TFRecord file.
        Currently available only for input data.
      FILE_LIST: Each line of the file is the location of an instance to
        process. Currently available only for input data.
      CSV: Values are comma-separated rows, with keys in a separate file.
        Currently available only for output data.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        JSON = 1
        TEXT = 2
        TF_RECORD = 3
        TF_RECORD_GZIP = 4
        FILE_LIST = 5
        CSV = 6

    class FrameworkValueValuesEnum(_messages.Enum):
        """Optional. The framework used to train this model. Only needed if
    model_version is a GCS path. Otherwise the framework specified during
    version creation will be used.

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

    class OutputDataFormatValueValuesEnum(_messages.Enum):
        """Optional. Format of the output data files, defaults to JSON.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format.
      JSON: Each line of the file is a JSON dictionary representing one
        record.
      TEXT: Deprecated. Use JSON instead.
      TF_RECORD: The source file is a TFRecord file. Currently available only
        for input data.
      TF_RECORD_GZIP: The source file is a GZIP-compressed TFRecord file.
        Currently available only for input data.
      FILE_LIST: Each line of the file is the location of an instance to
        process. Currently available only for input data.
      CSV: Values are comma-separated rows, with keys in a separate file.
        Currently available only for output data.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        JSON = 1
        TEXT = 2
        TF_RECORD = 3
        TF_RECORD_GZIP = 4
        FILE_LIST = 5
        CSV = 6
    accelerator = _messages.MessageField('GoogleCloudMlV1AcceleratorConfig', 1)
    batchSize = _messages.IntegerField(2)
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 3)
    framework = _messages.EnumField('FrameworkValueValuesEnum', 4)
    initialWorkerCount = _messages.IntegerField(5)
    inputPaths = _messages.StringField(6, repeated=True)
    maxWorkerCount = _messages.IntegerField(7)
    modelName = _messages.StringField(8)
    outputDataFormat = _messages.EnumField('OutputDataFormatValueValuesEnum', 9)
    outputPath = _messages.StringField(10)
    region = _messages.StringField(11)
    runtimeVersion = _messages.StringField(12)
    signatureName = _messages.StringField(13)
    tagsOverride = _messages.StringField(14, repeated=True)
    uri = _messages.StringField(15)
    versionName = _messages.StringField(16)
    workerType = _messages.StringField(17)