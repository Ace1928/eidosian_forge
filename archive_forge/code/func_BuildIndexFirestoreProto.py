from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def BuildIndexFirestoreProto(is_ancestor, properties):
    """Builds and returns a GoogleFirestoreAdminV1Index."""
    messages = firestore_utils.GetMessages()
    proto = messages.GoogleFirestoreAdminV1Index()
    proto.queryScope = COLLECTION_RECURSIVE if is_ancestor else COLLECTION_GROUP
    proto.apiScope = DATASTORE_API_SCOPE
    fields = []
    for prop in properties:
        field_proto = messages.GoogleFirestoreAdminV1IndexField()
        field_proto.fieldPath = prop.name
        if prop.direction == 'asc':
            field_proto.order = FIRESTORE_ASCENDING
        else:
            field_proto.order = FIRESTORE_DESCENDING
        fields.append(field_proto)
    proto.fields = fields
    return proto