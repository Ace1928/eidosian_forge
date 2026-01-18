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
class _SinglePropertyFilter(FilterPredicate):
    """Base class for a filter that operates on a single property."""

    def _get_prop_name(self):
        """Returns the name of the property being filtered."""
        raise NotImplementedError

    def _apply_to_value(self, value):
        """Apply the filter to the given value.

    Args:
      value: The comparable value to check.

    Returns:
      A boolean indicating if the given value matches the filter.
    """
        raise NotImplementedError

    def _get_prop_names(self):
        return set([self._get_prop_name()])

    def _apply(self, value_map):
        for other_value in value_map[self._get_prop_name()]:
            if self._apply_to_value(other_value):
                return True
        return False

    def _prune(self, value_map):
        if self._get_prop_name() not in value_map:
            return True
        values = [value for value in value_map[self._get_prop_name()] if self._apply_to_value(value)]
        value_map[self._get_prop_name()] = values
        return bool(values)