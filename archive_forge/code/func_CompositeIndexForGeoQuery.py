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
def CompositeIndexForGeoQuery(query):
    """Builds a descriptor for a composite index needed for a geo query.

  Args:
    query: A datastore_pb.Query instance.

  Returns:
    A tuple in the same form as produced by CompositeIndexForQuery.
  """
    required = True
    kind = query.kind()
    assert not query.has_ancestor()
    ancestor = False
    filters = query.filter_list()
    preintersection_props = set()
    geo_props = set()
    for filter in filters:
        name = filter.property(0).name()
        if filter.op() == datastore_pb.Query_Filter.EQUAL:
            preintersection_props.add(PropertySpec(name=name))
        else:
            assert filter.op() == datastore_pb.Query_Filter.CONTAINED_IN_REGION
            geo_props.add(PropertySpec(name=name, mode=GEOSPATIAL))
    prefix = frozenset(preintersection_props)
    postfix = (frozenset(geo_props),)
    return (required, kind, ancestor, (prefix, postfix))