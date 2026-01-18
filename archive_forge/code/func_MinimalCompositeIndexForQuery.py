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
def MinimalCompositeIndexForQuery(query, index_defs):
    """Computes the minimal composite index for this query.

  Unlike datastore_index.CompositeIndexForQuery, this function takes into
  account indexes that already exist in the system.

  Args:
    query: the datastore_pb.Query to compute suggestions for
    index_defs: a list of datastore_index.Index objects that already exist.

  Returns:
    None if no index is needed, otherwise the minimal index in the form
  (is_most_efficient, kind, ancestor, properties). Where is_most_efficient is a
  boolean denoting if the suggested index is the most efficient (i.e. the one
  returned by datastore_index.CompositeIndexForQuery). kind and ancestor
  are the same variables returned by datastore_index.CompositeIndexForQuery.
  properties is a tuple consisting of the prefix and postfix properties
  returend by datastore_index.CompositeIndexForQuery.
  """
    required, kind, ancestor, (prefix, postfix) = CompositeIndexForQuery(query)
    if not required:
        return None
    remaining_dict = {}
    for definition in index_defs:
        if kind != definition.kind or (not ancestor and definition.ancestor):
            continue
        _, _, index_props = IndexToKey(definition)
        index_prefix = _MatchPostfix(postfix, index_props)
        if index_prefix is None:
            continue
        remaining_index_props = set([prop.name for prop in index_prefix])
        if remaining_index_props - prefix:
            continue
        index_postfix = tuple(index_props[len(index_prefix):])
        remaining = remaining_dict.get(index_postfix)
        if remaining is None:
            remaining = (prefix.copy(), ancestor)
        props_remaining, ancestor_remaining = remaining
        props_remaining -= remaining_index_props
        if definition.ancestor:
            ancestor_remaining = False
        if not (props_remaining or ancestor_remaining):
            return None
        if (props_remaining, ancestor_remaining) == remaining:
            continue
        remaining_dict[index_postfix] = (props_remaining, ancestor_remaining)
    if not remaining_dict:
        return (True, kind, ancestor, (prefix, postfix))

    def calc_cost(minimal_props, minimal_ancestor):
        result = len(minimal_props)
        if minimal_ancestor:
            result += 2
        return result
    minimal_postfix, remaining = remaining_dict.popitem()
    minimal_props, minimal_ancestor = remaining
    minimal_cost = calc_cost(minimal_props, minimal_ancestor)
    for index_postfix, (props_remaining, ancestor_remaining) in remaining_dict.items():
        cost = calc_cost(props_remaining, ancestor_remaining)
        if cost < minimal_cost:
            minimal_cost = cost
            minimal_postfix = index_postfix
            minimal_props = props_remaining
            minimal_ancestor = ancestor_remaining
    props = (frozenset(minimal_props), (minimal_postfix, frozenset(), frozenset()))
    return (False, kind, minimal_ancestor, props)