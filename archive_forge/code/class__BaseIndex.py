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
class _BaseIndex(object):
    BUILDING, SERVING, DELETING, ERROR = range(4)
    ASCENDING = datastore_query.PropertyOrder.ASCENDING
    DESCENDING = datastore_query.PropertyOrder.DESCENDING

    def __init__(self, index_id, kind, has_ancestor, properties):
        """Construct a datastore index instance.

    Args:
      index_id: Required long; Uniquely identifies the index
      kind: Required string; Specifies the kind of the entities to index
      has_ancestor: Required boolean; indicates if the index supports a query
        that filters entities by the entity group parent
      properties: Required list of (string, int) tuples; The entity properties
        to index. First item in a tuple is the property name and the second
        item is the sorting direction (ASCENDING|DESCENDING).
        The order of the properties is based on the order in the index.
    """
        argument_error = datastore_errors.BadArgumentError
        datastore_types.ValidateInteger(index_id, 'index_id', argument_error, zero_ok=True)
        datastore_types.ValidateString(kind, 'kind', argument_error, empty_ok=True)
        if not isinstance(properties, (list, tuple)):
            raise argument_error('properties must be a list or a tuple')
        for idx, index_property in enumerate(properties):
            if not isinstance(index_property, (list, tuple)):
                raise argument_error('property[%d] must be a list or a tuple' % idx)
            if len(index_property) != 2:
                raise argument_error('property[%d] length should be 2 but was %d' % (idx, len(index_property)))
            datastore_types.ValidateString(index_property[0], 'property name', argument_error)
            _BaseIndex.__ValidateEnum(index_property[1], (self.ASCENDING, self.DESCENDING), 'sort direction')
        self.__id = long(index_id)
        self.__kind = kind
        self.__has_ancestor = bool(has_ancestor)
        self.__properties = properties

    @staticmethod
    def __ValidateEnum(value, accepted_values, name='value', exception=datastore_errors.BadArgumentError):
        datastore_types.ValidateInteger(value, name, exception)
        if not value in accepted_values:
            raise exception('%s should be one of %s but was %d' % (name, str(accepted_values), value))

    def _Id(self):
        """Returns the index id, a long."""
        return self.__id

    def _Kind(self):
        """Returns the index kind, a string.  Empty string ('') if none."""
        return self.__kind

    def _HasAncestor(self):
        """Indicates if this is an ancestor index, a boolean."""
        return self.__has_ancestor

    def _Properties(self):
        """Returns the index properties. a tuple of
    (index name as a string, [ASCENDING|DESCENDING]) tuples.
    """
        return self.__properties

    def __eq__(self, other):
        return self.__id == other.__id

    def __ne__(self, other):
        return self.__id != other.__id

    def __hash__(self):
        return hash(self.__id)