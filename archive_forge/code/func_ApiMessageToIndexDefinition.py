from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def ApiMessageToIndexDefinition(proto):
    """Converts a GoogleDatastoreAdminV1Index to an index definition structure."""
    properties = []
    for prop_proto in proto.properties:
        prop_definition = datastore_index.Property(name=str(prop_proto.name))
        if prop_proto.direction == DESCENDING:
            prop_definition.direction = 'desc'
        else:
            prop_definition.direction = 'asc'
        properties.append(prop_definition)
    index = datastore_index.Index(kind=str(proto.kind), properties=properties)
    if proto.ancestor is not NO_ANCESTOR:
        index.ancestor = True
    return (proto.indexId, index)