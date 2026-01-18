from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def CreateMissingIndexesViaFirestoreApi(project_id, database_id, index_definitions):
    """Creates the indexes via Firestore API if the index configuration is not present."""
    existing_indexes = ListDatastoreIndexesViaFirestoreApi(project_id, database_id)
    existing_indexes_normalized = NormalizeIndexes([index for _, index in existing_indexes])
    normalized_indexes = NormalizeIndexes(index_definitions.indexes)
    new_indexes = normalized_indexes - existing_indexes_normalized
    CreateIndexesViaFirestoreApi(project_id=project_id, database_id=database_id, indexes_to_create=new_indexes)