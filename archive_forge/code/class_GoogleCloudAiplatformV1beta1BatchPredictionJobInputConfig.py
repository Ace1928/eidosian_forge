from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchPredictionJobInputConfig(_messages.Message):
    """Configures the input to BatchPredictionJob. See
  Model.supported_input_storage_formats for Model's supported input formats,
  and how instances should be expressed via any of them.

  Fields:
    bigquerySource: The BigQuery location of the input table. The schema of
      the table should be in the format described by the given context OpenAPI
      Schema, if one is provided. The table may contain additional columns
      that are not described by the schema, and they will be ignored.
    gcsSource: The Cloud Storage location for the input instances.
    instancesFormat: Required. The format in which instances are given, must
      be one of the Model's supported_input_storage_formats.
  """
    bigquerySource = _messages.MessageField('GoogleCloudAiplatformV1beta1BigQuerySource', 1)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsSource', 2)
    instancesFormat = _messages.StringField(3)