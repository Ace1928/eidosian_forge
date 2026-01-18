from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesDropDatabaseRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesDropDatabaseRequest object.

  Fields:
    database: Required. The database to be dropped.
  """
    database = _messages.StringField(1, required=True)