import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
def _unserializer(registry, code, data):
    handler = registry.get(code)
    if not handler:
        return msgpack.ExtType(code, data)
    else:
        return handler.deserialize(data)