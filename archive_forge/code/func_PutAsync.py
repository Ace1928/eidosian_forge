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
def PutAsync(entities, **kwargs):
    """Asynchronously store one or more entities in the datastore.

  Identical to datastore.Put() except returns an asynchronous object. Call
  get_result() on the return value to block on the call and get the results.
  """
    extra_hook = kwargs.pop('extra_hook', None)
    config = _GetConfigFromKwargs(kwargs)
    if getattr(config, 'read_policy', None) == EVENTUAL_CONSISTENCY:
        raise datastore_errors.BadRequestError('read_policy is only supported on read operations.')
    entities, multiple = NormalizeAndTypeCheck(entities, Entity)
    for entity in entities:
        if entity.is_projection():
            raise datastore_errors.BadRequestError('Cannot put a partial entity: %s' % entity)
        if not entity.kind() or not entity.app():
            raise datastore_errors.BadRequestError('App and kind must not be empty, in entity: %s' % entity)

    def local_extra_hook(keys):
        num_keys = len(keys)
        num_entities = len(entities)
        if num_keys != num_entities:
            raise datastore_errors.InternalError('Put accepted %d entities but returned %d keys.' % (num_entities, num_keys))
        for entity, key in zip(entities, keys):
            if entity._Entity__key._Key__reference != key._Key__reference:
                assert not entity._Entity__key.has_id_or_name()
                entity._Entity__key._Key__reference.CopyFrom(key._Key__reference)
        if multiple:
            result = keys
        else:
            result = keys[0]
        if extra_hook:
            return extra_hook(result)
        return result
    return _GetConnection().async_put(config, entities, local_extra_hook)