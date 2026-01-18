from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataCacheConfig(_messages.Message):
    """Data cache is an optional feature available for Cloud SQL for MySQL
  Enterprise Plus edition only. For more information on data cache, see [Data
  cache overview](https://cloud.google.com/sql/help/mysql-data-cache) in Cloud
  SQL documentation.

  Fields:
    dataCacheEnabled: Optional. Whether data cache is enabled for the
      instance.
  """
    dataCacheEnabled = _messages.BooleanField(1)