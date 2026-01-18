from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
class _PropertyRangeFilter(_SinglePropertyFilter):
    """A filter predicate that represents a range of values.

  Since we allow multi-valued properties there is a large difference between
  "x > 0 AND x < 1" and "0 < x < 1." An entity with x = [-1, 2] will match the
  first but not the second.

  Since the datastore only allows a single inequality filter, multiple
  in-equality filters are merged into a single range filter in the
  datastore (unlike equality filters). This class is used by
  datastore_query.CompositeFilter to implement the same logic.
  """
    _start_key_value = None
    _end_key_value = None

    @datastore_rpc._positional(1)
    def __init__(self, start=None, start_incl=True, end=None, end_incl=True):
        """Constructs a range filter using start and end properties.

    Args:
      start: A entity_pb.Property to use as a lower bound or None to indicate
        no lower bound.
      start_incl: A boolean that indicates if the lower bound is inclusive.
      end: A entity_pb.Property to use as an upper bound or None to indicate
        no upper bound.
      end_incl: A boolean that indicates if the upper bound is inclusive.
    """
        if start is not None and (not isinstance(start, entity_pb.Property)):
            raise datastore_errors.BadArgumentError('start argument should be entity_pb.Property (%r)' % (start,))
        if end is not None and (not isinstance(end, entity_pb.Property)):
            raise datastore_errors.BadArgumentError('start argument should be entity_pb.Property (%r)' % (end,))
        if start and end and (start.name() != end.name()):
            raise datastore_errors.BadArgumentError('start and end arguments must be on the same property (%s != %s)' % (start.name(), end.name()))
        if not start and (not end):
            raise datastore_errors.BadArgumentError('Unbounded ranges are not supported.')
        super(_PropertyRangeFilter, self).__init__()
        self._start = start
        self._start_incl = start_incl
        self._end = end
        self._end_incl = end_incl

    @classmethod
    def from_property_filter(cls, prop_filter):
        op = prop_filter._filter.op()
        if op == datastore_pb.Query_Filter.GREATER_THAN:
            return cls(start=prop_filter._filter.property(0), start_incl=False)
        elif op == datastore_pb.Query_Filter.GREATER_THAN_OR_EQUAL:
            return cls(start=prop_filter._filter.property(0))
        elif op == datastore_pb.Query_Filter.LESS_THAN:
            return cls(end=prop_filter._filter.property(0), end_incl=False)
        elif op == datastore_pb.Query_Filter.LESS_THAN_OR_EQUAL:
            return cls(end=prop_filter._filter.property(0))
        else:
            raise datastore_errors.BadArgumentError('Unsupported operator (%s)' % (op,))

    def intersect(self, other):
        """Returns a filter representing the intersection of self and other."""
        if isinstance(other, PropertyFilter):
            other = self.from_property_filter(other)
        elif not isinstance(other, _PropertyRangeFilter):
            raise datastore_errors.BadArgumentError('other argument should be a _PropertyRangeFilter (%r)' % (other,))
        if other._get_prop_name() != self._get_prop_name():
            raise datastore_errors.BadArgumentError('other argument must be on the same property (%s != %s)' % (other._get_prop_name(), self._get_prop_name()))
        start_source = None
        if other._start:
            if self._start:
                result = cmp(self._get_start_key_value(), other._get_start_key_value())
                if result == 0:
                    result = cmp(other._start_incl, self._start_incl)
                if result > 0:
                    start_source = self
                elif result < 0:
                    start_source = other
            else:
                start_source = other
        elif self._start:
            start_source = self
        end_source = None
        if other._end:
            if self._end:
                result = cmp(self._get_end_key_value(), other._get_end_key_value())
                if result == 0:
                    result = cmp(self._end_incl, other._end_incl)
                if result < 0:
                    end_source = self
                elif result > 0:
                    end_source = other
            else:
                end_source = other
        elif self._end:
            end_source = self
        if start_source:
            if end_source in (start_source, None):
                return start_source
            result = _PropertyRangeFilter(start=start_source._start, start_incl=start_source._start_incl, end=end_source._end, end_incl=end_source._end_incl)
            result._start_key_value = start_source._start_key_value
            result._end_key_value = end_source._end_key_value
            return result
        else:
            return end_source or self

    def _get_start_key_value(self):
        if self._start_key_value is None:
            self._start_key_value = datastore_types.PropertyValueToKeyValue(self._start.value())
        return self._start_key_value

    def _get_end_key_value(self):
        if self._end_key_value is None:
            self._end_key_value = datastore_types.PropertyValueToKeyValue(self._end.value())
        return self._end_key_value

    def _apply_to_value(self, value):
        """Apply the filter to the given value.

    Args:
      value: The comparable value to check.

    Returns:
      A boolean indicating if the given value matches the filter.
    """
        if self._start:
            result = cmp(self._get_start_key_value(), value)
            if result > 0 or (result == 0 and (not self._start_incl)):
                return False
        if self._end:
            result = cmp(self._get_end_key_value(), value)
            if result < 0 or (result == 0 and (not self._end_incl)):
                return False
        return True

    def _get_prop_name(self):
        if self._start:
            return self._start.name()
        if self._end:
            return self._end.name()
        assert False

    def _to_pbs(self):
        pbs = []
        if self._start:
            if self._start_incl:
                op = datastore_pb.Query_Filter.GREATER_THAN_OR_EQUAL
            else:
                op = datastore_pb.Query_Filter.GREATER_THAN
            pb = datastore_pb.Query_Filter()
            pb.set_op(op)
            pb.add_property().CopyFrom(self._start)
            pbs.append(pb)
        if self._end:
            if self._end_incl:
                op = datastore_pb.Query_Filter.LESS_THAN_OR_EQUAL
            else:
                op = datastore_pb.Query_Filter.LESS_THAN
            pb = datastore_pb.Query_Filter()
            pb.set_op(op)
            pb.add_property().CopyFrom(self._end)
            pbs.append(pb)
        return pbs

    def _to_pb_v1(self, adapter):
        """Returns a googledatastore.Filter representation of the filter.

    Args:
      adapter: A datastore_rpc.AbstractAdapter.
    """
        filter_pb = googledatastore.Filter()
        composite_filter = filter_pb.composite_filter
        composite_filter.op = googledatastore.CompositeFilter.AND
        if self._start:
            if self._start_incl:
                op = googledatastore.PropertyFilter.GREATER_THAN_OR_EQUAL
            else:
                op = googledatastore.PropertyFilter.GREATER_THAN
            pb = composite_filter.filters.add().property_filter
            pb.op = op
            pb.property.name = self._start.name()
            adapter.get_entity_converter().v3_property_to_v1_value(self._start, True, pb.value)
        if self._end:
            if self._end_incl:
                op = googledatastore.PropertyFilter.LESS_THAN_OR_EQUAL
            else:
                op = googledatastore.PropertyFilter.LESS_THAN
            pb = composite_filter.filters.add().property_filter
            pb.op = op
            pb.property.name = self._end.name()
            adapter.get_entity_converter().v3_property_to_v1_value(self._end, True, pb.value)
        return filter_pb

    def __getstate__(self):
        raise pickle.PicklingError('Pickling of %r is unsupported.' % self)

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._start == other._start and self._end == other._end and (self._start_incl == other._start_incl or self._start is None) and (self._end_incl == other._end_incl or self._end is None)