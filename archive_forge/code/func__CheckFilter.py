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
def _CheckFilter(self, filter, values):
    """Type check a filter string and list of values.

    Raises BadFilterError if the filter string is empty, not a string, or
    invalid. Raises BadValueError if the value type is not supported.

    Args:
      filter: String containing the filter text.
      values: List of associated filter values.

    Returns:
      re.MatchObject (never None) that matches the 'filter'. Group 1 is the
      property name, group 3 is the operator. (Group 2 is unused.)
    """
    if isinstance(values, list) and (not values):
        raise datastore_errors.BadValueError('Cannot filter on []')
    try:
        match = Query.FILTER_REGEX.match(filter)
        if not match:
            raise datastore_errors.BadFilterError('Could not parse filter string: %s' % str(filter))
    except TypeError:
        raise datastore_errors.BadFilterError('Could not parse filter string: %s' % str(filter))
    property = match.group(1)
    operator = match.group(3)
    if operator is None:
        operator = '='
    if isinstance(values, tuple):
        values = list(values)
    elif not isinstance(values, list):
        values = [values]
    if isinstance(values[0], datastore_types._RAW_PROPERTY_TYPES):
        raise datastore_errors.BadValueError('Filtering on %s properties is not supported.' % typename(values[0]))
    if operator in self.INEQUALITY_OPERATORS and property != datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY:
        if self.__inequality_prop and property != self.__inequality_prop:
            raise datastore_errors.BadFilterError('Only one property per query may have inequality filters (%s).' % ', '.join(self.INEQUALITY_OPERATORS))
        elif len(self.__orderings) >= 1 and self.__orderings[0][0] != property:
            raise datastore_errors.BadFilterError('Inequality operators (%s) must be on the same property as the first sort order, if any sort orders are supplied' % ', '.join(self.INEQUALITY_OPERATORS))
    if self.__kind is None and property != datastore_types.KEY_SPECIAL_PROPERTY and (property != datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY):
        raise datastore_errors.BadFilterError('Only %s filters are allowed on kindless queries.' % datastore_types.KEY_SPECIAL_PROPERTY)
    if property == datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY:
        if self.__kind:
            raise datastore_errors.BadFilterError('Only kindless queries can have %s filters.' % datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY)
        if not operator in self.UPPERBOUND_INEQUALITY_OPERATORS:
            raise datastore_errors.BadFilterError('Only %s operators are supported with %s filters.' % (self.UPPERBOUND_INEQUALITY_OPERATORS, datastore_types._UNAPPLIED_LOG_TIMESTAMP_SPECIAL_PROPERTY))
    if property in datastore_types._SPECIAL_PROPERTIES:
        if property == datastore_types.KEY_SPECIAL_PROPERTY:
            for value in values:
                if not isinstance(value, Key):
                    raise datastore_errors.BadFilterError('%s filter value must be a Key; received %s (a %s)' % (datastore_types.KEY_SPECIAL_PROPERTY, value, typename(value)))
    return match