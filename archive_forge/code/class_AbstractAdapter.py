from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class AbstractAdapter(object):
    """Abstract interface between protobufs and user-level classes.

  This class defines conversions between the protobuf classes defined
  in entity_pb.py on the one hand, and the corresponding user-level
  classes (which are defined by higher-level API libraries such as
  datastore.py or db.py) on the other hand.

  The premise is that the code in this module is agnostic about the
  user-level classes used to represent keys and entities, while at the
  same time provinging APIs that accept or return such user-level
  classes.

  Higher-level libraries must subclass this abstract class and pass an
  instance of the subclass to the Connection they want to use.

  These methods may raise datastore_errors.Error for bad inputs.
  """
    _entity_converter = datastore_pbs.get_entity_converter()
    _query_converter = datastore_pbs._QueryConverter(_entity_converter)

    def __init__(self, id_resolver=None):
        if id_resolver:
            self._entity_converter = datastore_pbs.get_entity_converter(id_resolver)
            self._query_converter = datastore_pbs._QueryConverter(self._entity_converter)

    def get_entity_converter(self):
        return self._entity_converter

    def get_query_converter(self):
        return self._query_converter

    def pb_to_key(self, pb):
        """Turn an entity_pb.Reference into a user-level key."""
        raise NotImplementedError

    def pb_v1_to_key(self, pb):
        """Turn an googledatastore.Key into a user-level key."""
        v3_ref = entity_pb.Reference()
        self._entity_converter.v1_to_v3_reference(pb, v3_ref)
        return self.pb_to_key(v3_ref)

    def pb_to_entity(self, pb):
        """Turn an entity_pb.EntityProto into a user-level entity."""
        raise NotImplementedError

    def pb_v1_to_entity(self, pb, is_projection):
        """Turn an googledatastore.Entity into a user-level entity."""
        v3_entity = entity_pb.EntityProto()
        self._entity_converter.v1_to_v3_entity(pb, v3_entity, is_projection)
        return self.pb_to_entity(v3_entity)

    def pb_v1_to_query_result(self, pb, query_options):
        """Turn an googledatastore.Entity into a user-level query result."""
        if query_options.keys_only:
            return self.pb_v1_to_key(pb.key)
        else:
            return self.pb_v1_to_entity(pb, bool(query_options.projection))

    def pb_to_index(self, pb):
        """Turn an entity_pb.CompositeIndex into a user-level Index
    representation."""
        raise NotImplementedError

    def pb_to_query_result(self, pb, query_options):
        """Turn an entity_pb.EntityProto into a user-level query result."""
        if query_options.keys_only:
            return self.pb_to_key(pb.key())
        else:
            return self.pb_to_entity(pb)

    def key_to_pb(self, key):
        """Turn a user-level key into an entity_pb.Reference."""
        raise NotImplementedError

    def key_to_pb_v1(self, key):
        """Turn a user-level key into an googledatastore.Key."""
        v3_ref = self.key_to_pb(key)
        v1_key = googledatastore.Key()
        self._entity_converter.v3_to_v1_key(v3_ref, v1_key)
        return v1_key

    def entity_to_pb(self, entity):
        """Turn a user-level entity into an entity_pb.EntityProto."""
        raise NotImplementedError

    def entity_to_pb_v1(self, entity):
        """Turn a user-level entity into an googledatastore.Key."""
        v3_entity = self.entity_to_pb(entity)
        v1_entity = googledatastore.Entity()
        self._entity_converter.v3_to_v1_entity(v3_entity, v1_entity)
        return v1_entity

    def new_key_pb(self):
        """Create a new, empty entity_pb.Reference."""
        return entity_pb.Reference()

    def new_entity_pb(self):
        """Create a new, empty entity_pb.EntityProto."""
        return entity_pb.EntityProto()