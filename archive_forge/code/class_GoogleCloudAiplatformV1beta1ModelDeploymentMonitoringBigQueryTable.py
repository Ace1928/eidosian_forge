from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelDeploymentMonitoringBigQueryTable(_messages.Message):
    """ModelDeploymentMonitoringBigQueryTable specifies the BigQuery table name
  as well as some information of the logs stored in this table.

  Enums:
    LogSourceValueValuesEnum: The source of log.
    LogTypeValueValuesEnum: The type of log.

  Fields:
    bigqueryTablePath: The created BigQuery table to store logs. Customer
      could do their own query & analysis. Format:
      `bq://.model_deployment_monitoring_._`
    logSource: The source of log.
    logType: The type of log.
    requestResponseLoggingSchemaVersion: Output only. The schema version of
      the request/response logging BigQuery table. Default to v1 if unset.
  """

    class LogSourceValueValuesEnum(_messages.Enum):
        """The source of log.

    Values:
      LOG_SOURCE_UNSPECIFIED: Unspecified source.
      TRAINING: Logs coming from Training dataset.
      SERVING: Logs coming from Serving traffic.
    """
        LOG_SOURCE_UNSPECIFIED = 0
        TRAINING = 1
        SERVING = 2

    class LogTypeValueValuesEnum(_messages.Enum):
        """The type of log.

    Values:
      LOG_TYPE_UNSPECIFIED: Unspecified type.
      PREDICT: Predict logs.
      EXPLAIN: Explain logs.
    """
        LOG_TYPE_UNSPECIFIED = 0
        PREDICT = 1
        EXPLAIN = 2
    bigqueryTablePath = _messages.StringField(1)
    logSource = _messages.EnumField('LogSourceValueValuesEnum', 2)
    logType = _messages.EnumField('LogTypeValueValuesEnum', 3)
    requestResponseLoggingSchemaVersion = _messages.StringField(4)