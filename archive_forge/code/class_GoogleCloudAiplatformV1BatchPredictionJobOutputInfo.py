from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchPredictionJobOutputInfo(_messages.Message):
    """Further describes this job's output. Supplements output_config.

  Fields:
    bigqueryOutputDataset: Output only. The path of the BigQuery dataset
      created, in `bq://projectId.bqDatasetId` format, into which the
      prediction output is written.
    bigqueryOutputTable: Output only. The name of the BigQuery table created,
      in `predictions_` format, into which the prediction output is written.
      Can be used by UI to generate the BigQuery output path, for example.
    gcsOutputDirectory: Output only. The full path of the Cloud Storage
      directory created, into which the prediction output is written.
  """
    bigqueryOutputDataset = _messages.StringField(1)
    bigqueryOutputTable = _messages.StringField(2)
    gcsOutputDirectory = _messages.StringField(3)