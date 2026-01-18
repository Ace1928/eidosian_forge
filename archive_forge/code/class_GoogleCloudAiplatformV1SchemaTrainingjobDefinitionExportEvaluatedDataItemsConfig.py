from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionExportEvaluatedDataItemsConfig(_messages.Message):
    """Configuration for exporting test set predictions to a BigQuery table.

  Fields:
    destinationBigqueryUri: URI of desired destination BigQuery table.
      Expected format: `bq://{project_id}:{dataset_id}:{table}` If not
      specified, then results are exported to the following auto-created
      BigQuery table: `{project_id}:export_evaluated_examples_{model_name}_{yy
      yy_MM_dd'T'HH_mm_ss_SSS'Z'}.evaluated_examples`
    overrideExistingTable: If true and an export destination is specified,
      then the contents of the destination are overwritten. Otherwise, if the
      export destination already exists, then the export operation fails.
  """
    destinationBigqueryUri = _messages.StringField(1)
    overrideExistingTable = _messages.BooleanField(2)