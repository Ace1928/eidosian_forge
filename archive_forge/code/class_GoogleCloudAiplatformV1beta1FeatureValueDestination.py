from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureValueDestination(_messages.Message):
    """A destination location for Feature values and format.

  Fields:
    bigqueryDestination: Output in BigQuery format.
      BigQueryDestination.output_uri in
      FeatureValueDestination.bigquery_destination must refer to a table.
    csvDestination: Output in CSV format. Array Feature value types are not
      allowed in CSV format.
    tfrecordDestination: Output in TFRecord format. Below are the mapping from
      Feature value type in Featurestore to Feature value type in TFRecord:
      Value type in Featurestore | Value type in TFRecord DOUBLE, DOUBLE_ARRAY
      | FLOAT_LIST INT64, INT64_ARRAY | INT64_LIST STRING, STRING_ARRAY, BYTES
      | BYTES_LIST true -> byte_string("true"), false -> byte_string("false")
      BOOL, BOOL_ARRAY (true, false) | BYTES_LIST
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudAiplatformV1beta1BigQueryDestination', 1)
    csvDestination = _messages.MessageField('GoogleCloudAiplatformV1beta1CsvDestination', 2)
    tfrecordDestination = _messages.MessageField('GoogleCloudAiplatformV1beta1TFRecordDestination', 3)