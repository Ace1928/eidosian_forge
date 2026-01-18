from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def FirestoreApiMessageToIndexDefinition(proto):
    """Converts a GoogleFirestoreAdminV1Index to an index definition structure.

  Args:
    proto: GoogleFirestoreAdminV1Index

  Returns:
    index_id: A str to represent the index id in the resource path.
    index: A datastore_index.Index that contains index definition.

  Raises:
    ValueError: If GoogleFirestoreAdminV1Index cannot be converted to index
    definition structure.
  """
    properties = []
    for field_proto in proto.fields:
        prop_definition = datastore_index.Property(name=str(field_proto.fieldPath))
        if field_proto.order == FIRESTORE_DESCENDING:
            prop_definition.direction = 'desc'
        else:
            prop_definition.direction = 'asc'
        properties.append(prop_definition)
    collection_id, index_id = CollectionIdAndIndexIdFromResourcePath(proto.name)
    index = datastore_index.Index(kind=str(collection_id), properties=properties)
    if proto.apiScope != DATASTORE_API_SCOPE:
        raise ValueError('Invalid api scope: {}'.format(proto.apiScope))
    if proto.queryScope == COLLECTION_RECURSIVE:
        index.ancestor = True
    elif proto.queryScope == COLLECTION_GROUP:
        index.ancestor = False
    else:
        raise ValueError('Invalid query scope: {}'.format(proto.queryScope))
    return (index_id, index)