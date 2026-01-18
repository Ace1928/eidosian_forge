from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTablesResponse(_messages.Message):
    """Response message for
  google.bigtable.admin.v2.BigtableTableAdmin.ListTables

  Fields:
    nextPageToken: Set if not all tables could be returned in a single
      response. Pass this value to `page_token` in another request to get the
      next page of results.
    tables: The tables present in the requested instance.
  """
    nextPageToken = _messages.StringField(1)
    tables = _messages.MessageField('Table', 2, repeated=True)