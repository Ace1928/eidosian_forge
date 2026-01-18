from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def NormalizeIndexes(indexes):
    """Removes the last index property if it is __key__:asc which is redundant."""
    if not indexes:
        return set()
    for index in indexes:
        if index.properties and (index.properties[-1].name == '__key__' or index.properties[-1].name == '__name__') and (index.properties[-1].direction == 'asc'):
            index.properties.pop()
    return set(indexes)