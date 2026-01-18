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
def __InitConnection():
    """Internal method to make sure the connection state has been initialized."""
    if encoding.GetEncodedValue(os.environ, _ENV_KEY) and hasattr(_thread_local, 'connection_stack'):
        return

    def CreateConnection(adapter=None, _id_resolver=None, _api_version=datastore_rpc._DATASTORE_V3):
        if _id_resolver:
            adapter = DatastoreAdapter(_id_resolver=_id_resolver)
        return datastore_rpc.Connection(adapter=adapter, _api_version=_api_version)
    _thread_local.connection_stack = [datastore_rpc._CreateDefaultConnection(CreateConnection, adapter=_adapter)]
    os.environ[_ENV_KEY] = '1'