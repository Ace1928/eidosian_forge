from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def DeleteIndexes(project_id, indexes_to_delete_ids):
    """Sends the index deletion requests."""
    cnt = 0
    detail_message = None
    with progress_tracker.ProgressTracker('.', autotick=False, detail_message_callback=lambda: detail_message) as pt:
        for index_id in indexes_to_delete_ids:
            GetIndexesService().Delete(util.GetMessages().DatastoreProjectsIndexesDeleteRequest(projectId=project_id, indexId=index_id))
            cnt = cnt + 1
            detail_message = '{0:.0%}'.format(cnt / len(indexes_to_delete_ids))
            pt.Tick()