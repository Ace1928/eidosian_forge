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
def CompositeIndexForQuery(query):
    """Return the composite index needed for a query.

  A query is translated into a tuple, as follows:

  - The first item is the kind string, or None if we're not filtering
    on kind (see below).

  - The second item is a bool giving whether the query specifies an
    ancestor.

  - After that come (property, ASCENDING) pairs for those Filter
    entries whose operator is EQUAL or IN.  Since the order of these
    doesn't matter, they are sorted by property name to normalize them
    in order to avoid duplicates.

  - After that comes at most one (property, ASCENDING) pair for a
    Filter entry whose operator is on of the four inequalities.  There
    can be at most one of these.

  - After that come all the (property, direction) pairs for the Order
    entries, in the order given in the query.  Exceptions:
      (a) if there is a Filter entry with an inequality operator that matches
          the first Order entry, the first order pair is omitted (or,
          equivalently, in this case the inequality pair is omitted).
      (b) if an Order entry corresponds to an equality filter, it is ignored
          (since there will only ever be one value returned).
      (c) if there is an equality filter on __key__ all orders are dropped
          (since there will be at most one result returned).
      (d) if there is an order on __key__ all further orders are dropped (since
          keys are unique).
      (e) orders on __key__ ASCENDING are dropped (since this is supported
          natively by the datastore).

  - Finally, if there are Filter entries whose operator is EXISTS, and
    whose property names are not already listed, they are added, with
    the direction set to ASCENDING.

  This algorithm should consume all Filter and Order entries.

  Additional notes:

  - The low-level implementation allows queries that don't specify a
    kind; but the Python API doesn't support this yet.

  - If there's an inequality filter and one or more sort orders, the
    first sort order *must* match the inequality filter.

  - The following indexes are always built in and should be suppressed:
    - query on kind only;
    - query on kind and one filter *or* one order;
    - query on ancestor only, without kind (not exposed in Python yet);
    - query on kind and equality filters only, no order (with or without
      ancestor).

  - While the protocol buffer allows a Filter to contain multiple
    properties, we don't use this.  It is only needed for the IN operator
    but this is (currently) handled on the client side, so in practice
    each Filter is expected to have exactly one property.

  Args:
    query: A datastore_pb.Query instance.

  Returns:
    A tuple of the form (required, kind, ancestor, properties).
      required: boolean, whether the index is required;
      kind: the kind or None;
      ancestor: True if this is an ancestor query;
      properties: A tuple consisting of:
      - the prefix, represented by a set of property names
      - the postfix, represented by a tuple consisting of any number of:
        - Sets of property names or PropertySpec objects: these
          properties can appear in any order.
        - Sequences of PropertySpec objects: Indicates the properties
          must appear in the given order, with the specified direction (if
          specified in the PropertySpec).
  """
    required = True
    kind = query.kind()
    ancestor = query.has_ancestor()
    filters = query.filter_list()
    orders = query.order_list()
    for filter in filters:
        assert filter.op() != datastore_pb.Query_Filter.IN, 'Filter.op()==IN'
        nprops = len(filter.property_list())
        assert nprops == 1, 'Filter has %s properties, expected 1' % nprops
        if filter.op() == datastore_pb.Query_Filter.CONTAINED_IN_REGION:
            return CompositeIndexForGeoQuery(query)
    if not kind:
        required = False
    exists = list(query.property_name_list())
    exists.extend(query.group_by_property_name_list())
    filters, orders = RemoveNativelySupportedComponents(filters, orders, exists)
    eq_filters = [f for f in filters if f.op() in EQUALITY_OPERATORS]
    ineq_filters = [f for f in filters if f.op() in INEQUALITY_OPERATORS]
    exists_filters = [f for f in filters if f.op() in EXISTS_OPERATORS]
    assert len(eq_filters) + len(ineq_filters) + len(exists_filters) == len(filters), 'Not all filters used'
    if kind and (not ineq_filters) and (not exists_filters) and (not orders):
        names = set((f.property(0).name() for f in eq_filters))
        if not names.intersection(datastore_types._SPECIAL_PROPERTIES):
            required = False
    ineq_property = None
    if ineq_filters:
        for filter in ineq_filters:
            if filter.property(0).name() == datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY:
                continue
            if not ineq_property:
                ineq_property = filter.property(0).name()
            else:
                assert filter.property(0).name() == ineq_property
    group_by_props = set(query.group_by_property_name_list())
    prefix = frozenset((f.property(0).name() for f in eq_filters))
    postfix_ordered = [PropertySpec(name=order.property(), direction=order.direction()) for order in orders]
    postfix_group_by = frozenset((f.property(0).name() for f in exists_filters if f.property(0).name() in group_by_props))
    postfix_unordered = frozenset((f.property(0).name() for f in exists_filters if f.property(0).name() not in group_by_props))
    if ineq_property:
        if orders:
            assert ineq_property == orders[0].property()
        else:
            postfix_ordered.append(PropertySpec(name=ineq_property))
    property_count = len(prefix) + len(postfix_ordered) + len(postfix_group_by) + len(postfix_unordered)
    if kind and (not ancestor) and (property_count <= 1):
        required = False
        if postfix_ordered:
            prop = postfix_ordered[0]
            if prop.name == datastore_types.KEY_SPECIAL_PROPERTY and prop.direction == DESCENDING:
                required = True
    props = (prefix, (tuple(postfix_ordered), postfix_group_by, postfix_unordered))
    return (required, kind, ancestor, props)