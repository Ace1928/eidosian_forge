import time
import logging
import datetime
import functools
from pyzor.engines.common import *
class RedisDBHandle(BaseEngine):
    absolute_source = False
    handles_one_step = True
    log = logging.getLogger('pyzord')

    def __init__(self, fn, mode, max_age=None):
        self.max_age = max_age
        self._dsn = fn
        fn = fn.split(',')
        self.host = fn[0] or 'localhost'
        self.port = fn[1] or '6379'
        self.passwd = fn[2] or None
        self.db_name = fn[3] or '0'
        self.db = self._get_new_connection()
        self._check_version()

    @staticmethod
    def _encode_record(r):
        return {'r_count': r.r_count, 'r_entered': encode_date(r.r_entered), 'r_updated': encode_date(r.r_updated), 'wl_count': r.wl_count, 'wl_entered': encode_date(r.wl_entered), 'wl_updated': encode_date(r.wl_updated)}

    @staticmethod
    def _decode_record(r):
        if not r:
            return Record()
        return Record(r_count=int(r.get(b'r_count', 0)), r_entered=decode_date(r.get(b'r_entered', 0)), r_updated=decode_date(r.get(b'r_updated', 0)), wl_count=int(r.get(b'wl_count', 0)), wl_entered=decode_date(r.get(b'wl_entered', 0)), wl_updated=decode_date(r.get(b'wl_updated', 0)))

    def __iter__(self):
        for key in self.db.keys(self._real_key('*')):
            yield key.rsplit('.', 1)[-1]

    def _iteritems(self):
        for key in self:
            try:
                yield (key, self[key])
            except Exception as ex:
                self.log.warning('Invalid record %s: %s', key, ex)

    def iteritems(self):
        return self._iteritems()

    def items(self):
        return list(self._iteritems())

    @staticmethod
    def _real_key(key):
        return '%s.%s' % (NAMESPACE, key)

    @safe_call
    def _get_new_connection(self):
        if '/' in self.host:
            return redis.StrictRedis(unix_socket_path=self.host, db=int(self.db_name), password=self.passwd)
        return redis.StrictRedis(host=self.host, port=int(self.port), db=int(self.db_name), password=self.passwd)

    @safe_call
    def __getitem__(self, key):
        return self._decode_record(self.db.hgetall(self._real_key(key)))

    @safe_call
    def __setitem__(self, key, value):
        real_key = self._real_key(key)
        self.db.hmset(real_key, self._encode_record(value))
        if self.max_age is not None:
            self.db.expire(real_key, self.max_age)

    @safe_call
    def __delitem__(self, key):
        self.db.delete(self._real_key(key))

    @safe_call
    def report(self, keys):
        now = int(time.time())
        for key in keys:
            real_key = self._real_key(key)
            self.db.hincrby(real_key, 'r_count')
            self.db.hsetnx(real_key, 'r_entered', now)
            self.db.hset(real_key, 'r_updated', now)
            if self.max_age:
                self.db.expire(real_key, self.max_age)

    @safe_call
    def whitelist(self, keys):
        now = int(time.time())
        for key in keys:
            real_key = self._real_key(key)
            self.db.hincrby(real_key, 'wl_count')
            self.db.hsetnx(real_key, 'wl_entered', now)
            self.db.hset(real_key, 'wl_updated', now)
            if self.max_age:
                self.db.expire(real_key, self.max_age)

    @classmethod
    def get_prefork_connections(cls, fn, mode, max_age=None):
        """Yields a number of database connections suitable for a Pyzor
        pre-fork server.
        """
        while True:
            yield functools.partial(cls, fn, mode, max_age=max_age)

    def _check_version(self):
        """Check if there are deprecated records and warn the user."""
        old_keys = len(self.db.keys('pyzord.digest.*'))
        if old_keys:
            cmd = 'pyzor-migrate --delete --se=redis_v0 --sd=%s --de=redis --dd=%s' % (self._dsn, self._dsn)
            self.log.critical('You have %s records in the deprecated version of the redis engine.', old_keys)
            self.log.critical('Please migrate the records with: %r', cmd)