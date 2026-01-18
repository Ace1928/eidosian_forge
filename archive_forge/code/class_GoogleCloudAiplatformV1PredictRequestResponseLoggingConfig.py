from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PredictRequestResponseLoggingConfig(_messages.Message):
    """Configuration for logging request-response to a BigQuery table.

  Fields:
    bigqueryDestination: BigQuery table for logging. If only given a project,
      a new dataset will be created with name `logging__` where will be made
      BigQuery-dataset-name compatible (e.g. most special characters will
      become underscores). If no table name is given, a new table will be
      created with name `request_response_logging`
    enabled: If logging is enabled or not.
    samplingRate: Percentage of requests to be logged, expressed as a fraction
      in range(0,1].
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudAiplatformV1BigQueryDestination', 1)
    enabled = _messages.BooleanField(2)
    samplingRate = _messages.FloatField(3)