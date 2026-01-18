from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def BuildIndex(is_ancestor, kind, properties):
    """Builds and returns an index rep via GoogleDatastoreAdminV1Index."""
    index = datastore_index.Index(kind=str(kind), properties=[datastore_index.Property(name=str(prop[0]), direction=prop[1]) for prop in properties])
    index.ancestor = is_ancestor
    return index