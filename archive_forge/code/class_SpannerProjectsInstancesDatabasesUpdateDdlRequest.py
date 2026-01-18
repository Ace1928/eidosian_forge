from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesUpdateDdlRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesUpdateDdlRequest object.

  Fields:
    database: Required. The database to update.
    updateDatabaseDdlRequest: A UpdateDatabaseDdlRequest resource to be passed
      as the request body.
  """
    database = _messages.StringField(1, required=True)
    updateDatabaseDdlRequest = _messages.MessageField('UpdateDatabaseDdlRequest', 2)