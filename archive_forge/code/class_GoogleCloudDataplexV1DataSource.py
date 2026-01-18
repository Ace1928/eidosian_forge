from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataSource(_messages.Message):
    """The data source for DataScan.

  Fields:
    entity: Immutable. The Dataplex entity that represents the data source
      (e.g. BigQuery table) for DataScan, of the form: projects/{project_numbe
      r}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/entities/{ent
      ity_id}.
    resource: Immutable. The service-qualified full resource name of the cloud
      resource for a DataScan job to scan against. The field could be:
      BigQuery table of type "TABLE" for DataProfileScan/DataQualityScan
      Format: //bigquery.googleapis.com/projects/PROJECT_ID/datasets/DATASET_I
      D/tables/TABLE_ID
  """
    entity = _messages.StringField(1)
    resource = _messages.StringField(2)