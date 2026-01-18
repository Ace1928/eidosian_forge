from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataCatalogConfig(_messages.Message):
    """Specifies how metastore metadata should be integrated with the Data
  Catalog service.

  Fields:
    enabled: Optional. Defines whether the metastore metadata should be synced
      to Data Catalog. The default value is to disable syncing metastore
      metadata to Data Catalog.
  """
    enabled = _messages.BooleanField(1)