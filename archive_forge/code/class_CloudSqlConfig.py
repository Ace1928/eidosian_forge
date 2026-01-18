from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSqlConfig(_messages.Message):
    """Message for a Cloud SQL resource.

  Fields:
    settings: Settings for the Cloud SQL instance.
    version: The database version. e.g. "MYSQL_8_0". The version must match
      one of the values at https://cloud.google.com/sql/docs/mysql/admin-
      api/rest/v1beta4/SqlDatabaseVersion.
  """
    settings = _messages.MessageField('CloudSqlSettings', 1)
    version = _messages.StringField(2)