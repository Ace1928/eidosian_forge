import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class DateHandler(object):
    identity = 7
    handles = (datetime.date,)

    def __init__(self, registry):
        self._registry = registry

    def copy(self, registry):
        return type(self)(registry)

    def serialize(self, d):
        dct = {'year': d.year, 'month': d.month, 'day': d.day}
        return dumps(dct, registry=self._registry)

    def deserialize(self, blob):
        dct = loads(blob, registry=self._registry)
        if b'day' in dct:
            dct = dict(((k.decode('ascii'), v) for k, v in dct.items()))
        return datetime.date(year=dct['year'], month=dct['month'], day=dct['day'])