import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
class DateTimeHandler(object):
    identity = 1
    handles = (datetime.datetime,)

    def __init__(self, registry):
        self._registry = registry

    def copy(self, registry):
        return type(self)(registry)

    def serialize(self, dt):
        dct = {'day': dt.day, 'month': dt.month, 'year': dt.year, 'hour': dt.hour, 'minute': dt.minute, 'second': dt.second, 'microsecond': dt.microsecond}
        if dt.tzinfo:
            if zoneinfo:
                tz = str(dt.tzinfo)
            else:
                tz = dt.tzinfo.tzname(None)
            dct['tz'] = tz
        return dumps(dct, registry=self._registry)

    def deserialize(self, blob):
        dct = loads(blob, registry=self._registry)
        if b'day' in dct:
            dct = dict(((k.decode('ascii'), v) for k, v in dct.items()))
            if 'tz' in dct:
                dct['tz'] = dct['tz'].decode('ascii')
        dt = datetime.datetime(day=dct['day'], month=dct['month'], year=dct['year'], hour=dct['hour'], minute=dct['minute'], second=dct['second'], microsecond=dct['microsecond'])
        if 'tz' in dct and dct['tz']:
            if zoneinfo:
                tzinfo = zoneinfo.ZoneInfo(dct['tz'])
                dt = dt.replace(tzinfo=tzinfo)
            else:
                tzinfo = timezone(dct['tz'])
                dt = tzinfo.localize(dt)
        return dt