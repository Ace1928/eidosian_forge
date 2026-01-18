from __future__ import absolute_import
from ruamel import yaml
import copy
import itertools
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_object
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
def ProtoToIndexDefinition(proto):
    """Transform individual index protocol buffer to index definition.

  Args:
    proto: An instance of entity_pb.CompositeIndex to transform.

  Returns:
    A new instance of datastore_index.Index.
  """
    properties = []
    proto_index = proto.definition()
    for prop_proto in proto_index.property_list():
        prop_definition = Property(name=prop_proto.name())
        if prop_proto.mode() == entity_pb.Index_Property.GEOSPATIAL:
            prop_definition.mode = 'geospatial'
        elif prop_proto.direction() == entity_pb.Index_Property.DESCENDING:
            prop_definition.direction = 'desc'
        elif prop_proto.direction() == entity_pb.Index_Property.ASCENDING:
            prop_definition.direction = 'asc'
        properties.append(prop_definition)
    index = Index(kind=proto_index.entity_type(), properties=properties)
    if proto_index.ancestor():
        index.ancestor = True
    return index