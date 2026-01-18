from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesListRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesListRequest object.

  Fields:
    filter: A filter expression that filters conversion workspaces listed in
      the response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be either =, !=, >, or <. For example, list conversion workspaces
      created this year by specifying **createTime %gt;
      2020-01-01T00:00:00.000000000Z.** You can also filter nested fields. For
      example, you could specify **source.version = "12.c.1"** to select all
      conversion workspaces with source database version equal to 12.c.1.
    pageSize: The maximum number of conversion workspaces to return. The
      service may return fewer than this value. If unspecified, at most 50
      sets are returned.
    pageToken: The nextPageToken value received in the previous call to
      conversionWorkspaces.list, used in the subsequent request to retrieve
      the next page of results. On first call this should be left blank. When
      paginating, all other parameters provided to conversionWorkspaces.list
      must match the call that provided the page token.
    parent: Required. The parent which owns this collection of conversion
      workspaces.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)