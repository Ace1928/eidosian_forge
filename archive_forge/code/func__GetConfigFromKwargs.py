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
def _GetConfigFromKwargs(kwargs, convert_rpc=False, config_class=datastore_rpc.Configuration):
    """Get a Configuration object from the keyword arguments.

  This is purely an internal helper for the various public APIs below
  such as Get().

  Args:
    kwargs: A dict containing the keyword arguments passed to a public API.
    convert_rpc: If the an rpc should be converted or passed on directly.
    config_class: The config class that should be generated.

  Returns:
    A UserRPC instance, or a Configuration instance, or None.

  Raises:
    TypeError if unexpected keyword arguments are present.
  """
    if not kwargs:
        return None
    rpc = kwargs.pop('rpc', None)
    if rpc is not None:
        if not isinstance(rpc, apiproxy_stub_map.UserRPC):
            raise datastore_errors.BadArgumentError('rpc= argument should be None or a UserRPC instance')
        if 'config' in kwargs:
            raise datastore_errors.BadArgumentError('Expected rpc= or config= argument but not both')
        if not convert_rpc:
            if kwargs:
                raise datastore_errors.BadArgumentError('Unexpected keyword arguments: %s' % ', '.join(kwargs))
            return rpc
        read_policy = getattr(rpc, 'read_policy', None)
        kwargs['config'] = datastore_rpc.Configuration(deadline=rpc.deadline, read_policy=read_policy, config=_GetConnection().config)
    return config_class(**kwargs)