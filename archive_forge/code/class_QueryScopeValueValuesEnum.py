from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryScopeValueValuesEnum(_messages.Enum):
    """Indexes with a collection query scope specified allow queries against
    a collection that is the child of a specific document, specified at query
    time, and that has the same collection id. Indexes with a collection group
    query scope specified allow queries against all collections descended from
    a specific document, specified at query time, and that have the same
    collection id as this index.

    Values:
      QUERY_SCOPE_UNSPECIFIED: The query scope is unspecified. Not a valid
        option.
      COLLECTION: Indexes with a collection query scope specified allow
        queries against a collection that is the child of a specific document,
        specified at query time, and that has the collection id specified by
        the index.
      COLLECTION_GROUP: Indexes with a collection group query scope specified
        allow queries against all collections that has the collection id
        specified by the index.
      COLLECTION_RECURSIVE: Include all the collections's ancestor in the
        index. Only available for Datastore Mode databases.
    """
    QUERY_SCOPE_UNSPECIFIED = 0
    COLLECTION = 1
    COLLECTION_GROUP = 2
    COLLECTION_RECURSIVE = 3