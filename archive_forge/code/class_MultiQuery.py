import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
class MultiQuery(Query):
    """Class representing a query which requires multiple datastore queries.

  This class is actually a subclass of datastore.Query as it is intended to act
  like a normal Query object (supporting the same interface).

  Does not support keys only queries, since it needs whole entities in order
  to merge sort them. (That's not true if there are no sort orders, or if the
  sort order is on __key__, but allowing keys only queries in those cases, but
  not in others, would be confusing.)
  """

    def __init__(self, bound_queries, orderings):
        if len(bound_queries) > MAX_ALLOWABLE_QUERIES:
            raise datastore_errors.BadArgumentError('Cannot satisfy query -- too many subqueries (max: %d, got %d). Probable cause: too many IN/!= filters in query.' % (MAX_ALLOWABLE_QUERIES, len(bound_queries)))
        projection = bound_queries and bound_queries[0].GetQueryOptions().projection
        for query in bound_queries:
            if projection != query.GetQueryOptions().projection:
                raise datastore_errors.BadQueryError('All queries must have the same projection.')
            if query.IsKeysOnly():
                raise datastore_errors.BadQueryError('MultiQuery does not support keys_only.')
        self.__projection = projection
        self.__bound_queries = bound_queries
        self.__orderings = orderings
        self.__compile = False

    def __str__(self):
        res = 'MultiQuery: '
        for query in self.__bound_queries:
            res = '%s %s' % (res, str(query))
        return res

    def Get(self, limit, offset=0, **kwargs):
        """Deprecated, use list(Run(...)) instead.

    Args:
      limit: int or long representing the maximum number of entities to return.
      offset: int or long representing the number of entities to skip
      kwargs: Any keyword arguments accepted by datastore_query.QueryOptions().

    Returns:
      A list of entities with at most "limit" entries (less if the query
      completes before reading limit values).
    """
        if limit is None:
            kwargs.setdefault('batch_size', _MAX_INT_32)
        return list(self.Run(limit=limit, offset=offset, **kwargs))

    class SortOrderEntity(object):
        """Allow entity comparisons using provided orderings.

    The iterator passed to the constructor is eventually consumed via
    calls to GetNext(), which generate new SortOrderEntity s with the
    same orderings.
    """

        def __init__(self, entity_iterator, orderings):
            """Ctor.

      Args:
        entity_iterator: an iterator of entities which will be wrapped.
        orderings: an iterable of (identifier, order) pairs. order
          should be either Query.ASCENDING or Query.DESCENDING.
      """
            self.__entity_iterator = entity_iterator
            self.__entity = None
            self.__min_max_value_cache = {}
            try:
                self.__entity = entity_iterator.next()
            except StopIteration:
                pass
            else:
                self.__orderings = orderings

        def __str__(self):
            return str(self.__entity)

        def GetEntity(self):
            """Gets the wrapped entity."""
            return self.__entity

        def GetNext(self):
            """Wrap and return the next entity.

      The entity is retrieved from the iterator given at construction time.
      """
            return MultiQuery.SortOrderEntity(self.__entity_iterator, self.__orderings)

        def CmpProperties(self, that):
            """Compare two entities and return their relative order.

      Compares self to that based on the current sort orderings and the
      key orders between them. Returns negative, 0, or positive depending on
      whether self is less, equal to, or greater than that. This
      comparison returns as if all values were to be placed in ascending order
      (highest value last).  Only uses the sort orderings to compare (ignores
       keys).

      Args:
        that: SortOrderEntity

      Returns:
        Negative if self < that
        Zero if self == that
        Positive if self > that
      """
            if not self.__entity:
                return cmp(self.__entity, that.__entity)
            for identifier, order in self.__orderings:
                value1 = self.__GetValueForId(self, identifier, order)
                value2 = self.__GetValueForId(that, identifier, order)
                result = cmp(value1, value2)
                if order == Query.DESCENDING:
                    result = -result
                if result:
                    return result
            return 0

        def __GetValueForId(self, sort_order_entity, identifier, sort_order):
            value = _GetPropertyValue(sort_order_entity.__entity, identifier)
            if isinstance(value, list):
                entity_key = sort_order_entity.__entity.key()
                if (entity_key, identifier) in self.__min_max_value_cache:
                    value = self.__min_max_value_cache[entity_key, identifier]
                elif sort_order == Query.DESCENDING:
                    value = min(value)
                else:
                    value = max(value)
                self.__min_max_value_cache[entity_key, identifier] = value
            return value

        def __cmp__(self, that):
            """Compare self to that w.r.t. values defined in the sort order.

      Compare an entity with another, using sort-order first, then the key
      order to break ties. This can be used in a heap to have faster min-value
      lookup.

      Args:
        that: other entity to compare to
      Returns:
        negative: if self is less than that in sort order
        zero: if self is equal to that in sort order
        positive: if self is greater than that in sort order
      """
            property_compare = self.CmpProperties(that)
            if property_compare:
                return property_compare
            else:
                return cmp(self.__entity.key(), that.__entity.key())

    def _ExtractBounds(self, config):
        """This function extracts the range of results to consider.

    Since MultiQuery dedupes in memory, we must apply the offset and limit in
    memory. The results that should be considered are
    results[lower_bound:upper_bound].

    We also pass the offset=0 and limit=upper_bound to the base queries to
    optimize performance.

    Args:
      config: The base datastore_query.QueryOptions.

    Returns:
      a tuple consisting of the lower_bound and upper_bound to impose in memory
      and the config to use with each bound query. The upper_bound may be None.
    """
        if config is None:
            return (0, None, None)
        lower_bound = config.offset or 0
        upper_bound = config.limit
        if lower_bound:
            if upper_bound is not None:
                upper_bound = min(lower_bound + upper_bound, _MAX_INT_32)
            config = datastore_query.QueryOptions(offset=0, limit=upper_bound, config=config)
        return (lower_bound, upper_bound, config)

    def __GetProjectionOverride(self, config):
        """Returns a tuple of (original projection, projection override).

    If projection is None, there is no projection. If override is None,
    projection is sufficent for this query.
    """
        projection = datastore_query.QueryOptions.projection(config)
        if projection is None:
            projection = self.__projection
        else:
            projection = projection
        if not projection:
            return (None, None)
        override = set()
        for prop, _ in self.__orderings:
            if prop not in projection:
                override.add(prop)
        if not override:
            return (projection, None)
        return (projection, projection + tuple(override))

    def Run(self, **kwargs):
        """Return an iterable output with all results in order.

    Merge sort the results. First create a list of iterators, then walk
    though them and yield results in order.

    Args:
      kwargs: Any keyword arguments accepted by datastore_query.QueryOptions().

    Returns:
      An iterator for the result set.
    """
        config = _GetConfigFromKwargs(kwargs, convert_rpc=True, config_class=datastore_query.QueryOptions)
        if config and config.keys_only:
            raise datastore_errors.BadRequestError('keys only queries are not supported by multi-query.')
        lower_bound, upper_bound, config = self._ExtractBounds(config)
        projection, override = self.__GetProjectionOverride(config)
        if override:
            config = datastore_query.QueryOptions(projection=override, config=config)
        results = []
        count = 1
        log_level = logging.DEBUG - 1
        for bound_query in self.__bound_queries:
            logging.log(log_level, 'Running query #%i' % count)
            results.append(bound_query.Run(config=config))
            count += 1

        def GetDedupeKey(sort_order_entity):
            if projection:
                return (sort_order_entity.GetEntity().key(), frozenset(sort_order_entity.GetEntity().iteritems()))
            else:
                return sort_order_entity.GetEntity().key()

        def IterateResults(results):
            """Iterator function to return all results in sorted order.

      Iterate over the array of results, yielding the next element, in
      sorted order. This function is destructive (results will be empty
      when the operation is complete).

      Args:
        results: list of result iterators to merge and iterate through

      Yields:
        The next result in sorted order.
      """
            result_heap = []
            for result in results:
                heap_value = MultiQuery.SortOrderEntity(result, self.__orderings)
                if heap_value.GetEntity():
                    heapq.heappush(result_heap, heap_value)
            used_keys = set()
            while result_heap:
                if upper_bound is not None and len(used_keys) >= upper_bound:
                    break
                top_result = heapq.heappop(result_heap)
                dedupe_key = GetDedupeKey(top_result)
                if dedupe_key not in used_keys:
                    result = top_result.GetEntity()
                    if override:
                        for key in result.keys():
                            if key not in projection:
                                del result[key]
                    yield result
                else:
                    pass
                used_keys.add(dedupe_key)
                results_to_push = []
                while result_heap:
                    next = heapq.heappop(result_heap)
                    if dedupe_key != GetDedupeKey(next):
                        results_to_push.append(next)
                        break
                    else:
                        results_to_push.append(next.GetNext())
                results_to_push.append(top_result.GetNext())
                for popped_result in results_to_push:
                    if popped_result.GetEntity():
                        heapq.heappush(result_heap, popped_result)
        it = IterateResults(results)
        try:
            for _ in xrange(lower_bound):
                it.next()
        except StopIteration:
            pass
        return it

    def Count(self, limit=1000, **kwargs):
        """Return the number of matched entities for this query.

    Will return the de-duplicated count of results.  Will call the more
    efficient Get() function if a limit is given.

    Args:
      limit: maximum number of entries to count (for any result > limit, return
      limit).
      config: Optional Configuration to use for this request.

    Returns:
      count of the number of entries returned.
    """
        kwargs['limit'] = limit
        config = _GetConfigFromKwargs(kwargs, convert_rpc=True, config_class=datastore_query.QueryOptions)
        projection, override = self.__GetProjectionOverride(config)
        if not projection:
            config = datastore_query.QueryOptions(keys_only=True, config=config)
        elif override:
            config = datastore_query.QueryOptions(projection=override, config=config)
        lower_bound, upper_bound, config = self._ExtractBounds(config)
        used_keys = set()
        for bound_query in self.__bound_queries:
            for result in bound_query.Run(config=config):
                if projection:
                    dedupe_key = (result.key(), tuple(result.iteritems()))
                else:
                    dedupe_key = result
                used_keys.add(dedupe_key)
                if upper_bound and len(used_keys) >= upper_bound:
                    return upper_bound - lower_bound
        return max(0, len(used_keys) - lower_bound)

    def GetIndexList(self):
        raise AssertionError('No index_list available for a MultiQuery (queries using "IN" or "!=" operators)')

    def GetCursor(self):
        raise AssertionError('No cursor available for a MultiQuery (queries using "IN" or "!=" operators)')

    def _GetCompiledQuery(self):
        """Internal only, do not use."""
        raise AssertionError('No compilation available for a MultiQuery (queries using "IN" or "!=" operators)')

    def __setitem__(self, query_filter, value):
        """Add a new filter by setting it on all subqueries.

    If any of the setting operations raise an exception, the ones
    that succeeded are undone and the exception is propagated
    upward.

    Args:
      query_filter: a string of the form "property operand".
      value: the value that the given property is compared against.
    """
        saved_items = []
        for index, query in enumerate(self.__bound_queries):
            saved_items.append(query.get(query_filter, None))
            try:
                query[query_filter] = value
            except:
                for q, old_value in itertools.izip(self.__bound_queries[:index], saved_items):
                    if old_value is not None:
                        q[query_filter] = old_value
                    else:
                        del q[query_filter]
                raise

    def __delitem__(self, query_filter):
        """Delete a filter by deleting it from all subqueries.

    If a KeyError is raised during the attempt, it is ignored, unless
    every subquery raised a KeyError. If any other exception is
    raised, any deletes will be rolled back.

    Args:
      query_filter: the filter to delete.

    Raises:
      KeyError: No subquery had an entry containing query_filter.
    """
        subquery_count = len(self.__bound_queries)
        keyerror_count = 0
        saved_items = []
        for index, query in enumerate(self.__bound_queries):
            try:
                saved_items.append(query.get(query_filter, None))
                del query[query_filter]
            except KeyError:
                keyerror_count += 1
            except:
                for q, old_value in itertools.izip(self.__bound_queries[:index], saved_items):
                    if old_value is not None:
                        q[query_filter] = old_value
                raise
        if keyerror_count == subquery_count:
            raise KeyError(query_filter)

    def __iter__(self):
        return iter(self.__bound_queries)
    GetCompiledCursor = GetCursor
    GetCompiledQuery = _GetCompiledQuery