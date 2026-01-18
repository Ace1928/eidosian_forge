from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlFlagsListRequest(_messages.Message):
    """A SqlFlagsListRequest object.

  Fields:
    databaseVersion: Database type and version you want to retrieve flags for.
      By default, this method returns flags for all database types and
      versions.
  """
    databaseVersion = _messages.StringField(1)