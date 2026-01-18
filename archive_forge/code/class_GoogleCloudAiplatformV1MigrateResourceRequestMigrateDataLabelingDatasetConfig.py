from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1MigrateResourceRequestMigrateDataLabelingDatasetConfig(_messages.Message):
    """Config for migrating Dataset in datalabeling.googleapis.com to Vertex
  AI's Dataset.

  Fields:
    dataset: Required. Full resource name of data labeling Dataset. Format:
      `projects/{project}/datasets/{dataset}`.
    datasetDisplayName: Optional. Display name of the Dataset in Vertex AI.
      System will pick a display name if unspecified.
    migrateDataLabelingAnnotatedDatasetConfigs: Optional. Configs for
      migrating AnnotatedDataset in datalabeling.googleapis.com to Vertex AI's
      SavedQuery. The specified AnnotatedDatasets have to belong to the
      datalabeling Dataset.
  """
    dataset = _messages.StringField(1)
    datasetDisplayName = _messages.StringField(2)
    migrateDataLabelingAnnotatedDatasetConfigs = _messages.MessageField('GoogleCloudAiplatformV1MigrateResourceRequestMigrateDataLabelingDatasetConfigMigrateDataLabelingAnnotatedDatasetConfig', 3, repeated=True)