from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def CreateMissingIndexes(project_id, index_definitions):
    """Creates the indexes if the index configuration is not present."""
    indexes = ListIndexes(project_id)
    normalized_indexes = NormalizeIndexes(index_definitions.indexes)
    new_indexes = normalized_indexes - {index for _, index in indexes}
    CreateIndexes(project_id, new_indexes)