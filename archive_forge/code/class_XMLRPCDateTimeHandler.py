import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class XMLRPCDateTimeHandler(object):
    handles = (xmlrpclib.DateTime,)
    identity = 6

    def __init__(self, registry):
        self._handler = DateTimeHandler(registry)

    def copy(self, registry):
        return type(self)(registry)

    def serialize(self, obj):
        dt = datetime.datetime(*tuple(obj.timetuple())[:6])
        return self._handler.serialize(dt)

    def deserialize(self, blob):
        dt = self._handler.deserialize(blob)
        return xmlrpclib.DateTime(dt.timetuple())