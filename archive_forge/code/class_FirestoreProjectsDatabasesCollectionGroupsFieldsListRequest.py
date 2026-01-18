from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCollectionGroupsFieldsListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCollectionGroupsFieldsListRequest object.

  Fields:
    filter: The filter to apply to list results. Currently,
      FirestoreAdmin.ListFields only supports listing fields that have been
      explicitly overridden. To issue this query, call
      FirestoreAdmin.ListFields with a filter that includes
      `indexConfig.usesAncestorConfig:false` .
    pageSize: The number of results to return.
    pageToken: A page token, returned from a previous call to
      FirestoreAdmin.ListFields, that may be used to get the next page of
      results.
    parent: Required. A parent name of the form `projects/{project_id}/databas
      es/{database_id}/collectionGroups/{collection_id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)