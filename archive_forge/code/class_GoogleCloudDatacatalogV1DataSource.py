from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataSource(_messages.Message):
    """Physical location of an entry.

  Enums:
    ServiceValueValuesEnum: Service that physically stores the data.

  Fields:
    resource: Full name of a resource as defined by the service. For example:
      `//bigquery.googleapis.com/projects/{PROJECT_ID}/locations/{LOCATION}/da
      tasets/{DATASET_ID}/tables/{TABLE_ID}`
    service: Service that physically stores the data.
    sourceEntry: Output only. Data Catalog entry name, if applicable.
    storageProperties: Detailed properties of the underlying storage.
  """

    class ServiceValueValuesEnum(_messages.Enum):
        """Service that physically stores the data.

    Values:
      SERVICE_UNSPECIFIED: Default unknown service.
      CLOUD_STORAGE: Google Cloud Storage service.
      BIGQUERY: BigQuery service.
    """
        SERVICE_UNSPECIFIED = 0
        CLOUD_STORAGE = 1
        BIGQUERY = 2
    resource = _messages.StringField(1)
    service = _messages.EnumField('ServiceValueValuesEnum', 2)
    sourceEntry = _messages.StringField(3)
    storageProperties = _messages.MessageField('GoogleCloudDatacatalogV1StorageProperties', 4)