from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def BuildIndexProto(ancestor, kind, project_id, properties):
    """Builds and returns a GoogleDatastoreAdminV1Index."""
    messages = util.GetMessages()
    proto = messages.GoogleDatastoreAdminV1Index()
    proto.projectId = project_id
    proto.kind = kind
    proto.ancestor = ancestor
    proto.state = CREATING
    props = []
    for prop in properties:
        prop_proto = messages.GoogleDatastoreAdminV1IndexedProperty()
        prop_proto.name = prop.name
        if prop.direction == 'asc':
            prop_proto.direction = ASCENDING
        else:
            prop_proto.direction = DESCENDING
        props.append(prop_proto)
    proto.properties = props
    return proto