from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelMonitoringConfig(_messages.Message):
    """The model monitoring configuration used for Batch Prediction Job.

  Fields:
    alertConfig: Model monitoring alert config.
    analysisInstanceSchemaUri: YAML schema file uri in Cloud Storage
      describing the format of a single instance that you want Tensorflow Data
      Validation (TFDV) to analyze. If there are any data type differences
      between predict instance and TFDV instance, this field can be used to
      override the schema. For models trained with Vertex AI, this field must
      be set as all the fields in predict instance formatted as string.
    objectiveConfigs: Model monitoring objective config.
    statsAnomaliesBaseDirectory: A Google Cloud Storage location for batch
      prediction model monitoring to dump statistics and anomalies. If not
      provided, a folder will be created in customer project to hold
      statistics and anomalies.
  """
    alertConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringAlertConfig', 1)
    analysisInstanceSchemaUri = _messages.StringField(2)
    objectiveConfigs = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfig', 3, repeated=True)
    statsAnomaliesBaseDirectory = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsDestination', 4)