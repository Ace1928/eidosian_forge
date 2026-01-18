from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataplexSpec(_messages.Message):
    """Common Dataplex fields.

  Fields:
    asset: Fully qualified resource name of an asset in Dataplex, to which the
      underlying data source (Cloud Storage bucket or BigQuery dataset) of the
      entity is attached.
    compressionFormat: Compression format of the data, e.g., zip, gzip etc.
    dataFormat: Format of the data.
    projectId: Project ID of the underlying Cloud Storage or BigQuery data.
      Note that this may not be the same project as the correspondingly
      Dataplex lake / zone / asset.
  """
    asset = _messages.StringField(1)
    compressionFormat = _messages.StringField(2)
    dataFormat = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchema', 3)
    projectId = _messages.StringField(4)