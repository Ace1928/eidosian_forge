from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def CreateIndexesViaFirestoreApi(project_id, database_id, indexes_to_create):
    """Sends the index creation requests via Firestore API."""
    detail_message = None
    with progress_tracker.ProgressTracker('.', autotick=False, detail_message_callback=lambda: detail_message) as pt:
        for i, index in enumerate(indexes_to_create):
            firestore_indexes.CreateIndex(project_id, database_id, index.kind, BuildIndexFirestoreProto(index.ancestor, index.properties))
            detail_message = '{0:.0%}'.format(i / len(indexes_to_create))
            pt.Tick()