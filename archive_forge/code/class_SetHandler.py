import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class SetHandler(object):
    identity = 4
    handles = (set,)

    def __init__(self, registry):
        self._registry = registry

    def copy(self, registry):
        return type(self)(registry)

    def serialize(self, obj):
        return dumps(list(obj), registry=self._registry)

    def deserialize(self, data):
        return self.handles[0](loads(data, registry=self._registry))