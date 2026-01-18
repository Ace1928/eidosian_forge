import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
class MySQLDBHandle(BaseEngine):
    absolute_source = False
    handles_one_step = True
    reorganize_period = 3600 * 24
    reconnect_period = 60
    log = logging.getLogger('pyzord')

    def __init__(self, fn, mode, max_age=None):
        self.max_age = max_age
        self.db = None
        self.host, self.user, self.passwd, self.db_name, self.table_name = fn.split(',')
        self.last_connect_attempt = 0
        self.reorganize_timer = None
        self.reconnect()
        self.start_reorganizing()

    def _get_new_connection(self):
        """Returns a new db connection."""
        db = MySQLdb.connect(host=self.host, user=self.user, db=self.db_name, passwd=self.passwd)
        db.autocommit(True)
        return db

    def _check_reconnect_time(self):
        if time.time() - self.last_connect_attempt < self.reconnect_period:
            self.log.debug("Can't reconnect until %s", time.ctime(self.last_connect_attempt + self.reconnect_period))
            return False
        return True

    def reconnect(self):
        if not self._check_reconnect_time():
            return
        if self.db:
            try:
                self.db.close()
            except MySQLdb.Error:
                pass
        try:
            self.db = self._get_new_connection()
        except MySQLdb.Error as e:
            self.log.error('Unable to connect to database: %s', e)
            self.db = None
        self.last_connect_attempt = time.time()

    def _iter(self, db):
        c = db.cursor(cursorclass=MySQLdb.cursors.SSCursor)
        c.execute('SELECT digest FROM %s' % self.table_name)
        while True:
            row = c.fetchone()
            if not row:
                break
            yield row[0]
        c.close()

    def __iter__(self):
        return self._safe_call('iter', self._iter, ())

    def _iteritems(self, db):
        c = db.cursor(cursorclass=MySQLdb.cursors.SSCursor)
        c.execute('SELECT digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated FROM %s' % self.table_name)
        while True:
            row = c.fetchone()
            if not row:
                break
            yield (row[0], Record(*row[1:]))
        c.close()

    def iteritems(self):
        return self._safe_call('iteritems', self._iteritems, ())

    def items(self):
        return list(self._safe_call('iteritems', self._iteritems, ()))

    def __del__(self):
        """Close the database when the object is no longer needed."""
        try:
            if self.db:
                self.db.close()
        except MySQLdb.Error:
            pass

    def _safe_call(self, name, method, args):
        try:
            return method(*args, db=self.db)
        except (MySQLdb.Error, AttributeError) as ex:
            self.log.error('%s failed: %s', name, ex)
            self.reconnect()
            raise DatabaseError('Database temporarily unavailable.')

    def report(self, keys):
        return self._safe_call('report', self._report, (keys,))

    def whitelist(self, keys):
        return self._safe_call('whitelist', self._whitelist, (keys,))

    def __getitem__(self, key):
        return self._safe_call('getitem', self._really__getitem__, (key,))

    def __setitem__(self, key, value):
        return self._safe_call('setitem', self._really__setitem__, (key, value))

    def __delitem__(self, key):
        return self._safe_call('delitem', self._really__delitem__, (key,))

    def _report(self, keys, db=None):
        c = db.cursor()
        try:
            c.executemany('INSERT INTO %s (digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated) VALUES (%%s, 1, 0, NOW(), NOW(), NOW(), NOW()) ON DUPLICATE KEY UPDATE r_count=r_count+1, r_updated=NOW()' % self.table_name, itertools.imap(lambda key: (key,), keys))
        finally:
            c.close()

    def _whitelist(self, keys, db=None):
        c = db.cursor()
        try:
            c.executemany('INSERT INTO %s (digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated) VALUES (%%s, 0, 1, NOW(), NOW(), NOW(), NOW()) ON DUPLICATE KEY UPDATE wl_count=wl_count+1, wl_updated=NOW()' % self.table_name, itertools.imap(lambda key: (key,), keys))
        finally:
            c.close()

    def _really__getitem__(self, key, db=None):
        """__getitem__ without the exception handling."""
        c = db.cursor()
        c.execute('SELECT r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated FROM %s WHERE digest=%%s' % self.table_name, (key,))
        try:
            try:
                return Record(*c.fetchone())
            except TypeError:
                raise KeyError()
        finally:
            c.close()

    def _really__setitem__(self, key, value, db=None):
        """__setitem__ without the exception handling."""
        c = db.cursor()
        try:
            c.execute('INSERT INTO %s (digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated) VALUES (%%s, %%s, %%s, %%s, %%s, %%s, %%s) ON DUPLICATE KEY UPDATE r_count=%%s, wl_count=%%s, r_entered=%%s, r_updated=%%s, wl_entered=%%s, wl_updated=%%s' % self.table_name, (key, value.r_count, value.wl_count, value.r_entered, value.r_updated, value.wl_entered, value.wl_updated, value.r_count, value.wl_count, value.r_entered, value.r_updated, value.wl_entered, value.wl_updated))
        finally:
            c.close()

    def _really__delitem__(self, key, db=None):
        """__delitem__ without the exception handling."""
        c = db.cursor()
        try:
            c.execute('DELETE FROM %s WHERE digest=%%s' % self.table_name, (key,))
        finally:
            c.close()

    def start_reorganizing(self):
        if not self.max_age:
            return
        self.log.debug('reorganizing the database')
        breakpoint = datetime.datetime.now() - datetime.timedelta(seconds=self.max_age)
        db = self._get_new_connection()
        c = db.cursor()
        try:
            c.execute('DELETE FROM %s WHERE r_updated<%%s' % self.table_name, (breakpoint,))
        except (MySQLdb.Error, AttributeError) as e:
            self.log.warn('Unable to reorganise: %s', e)
        finally:
            c.close()
            db.close()
        self.reorganize_timer = threading.Timer(self.reorganize_period, self.start_reorganizing)
        self.reorganize_timer.setDaemon(True)
        self.reorganize_timer.start()

    @classmethod
    def get_prefork_connections(cls, fn, mode, max_age=None):
        """Yields a number of database connections suitable for a Pyzor
        pre-fork server.
        """
        yield functools.partial(cls, fn, mode, max_age=max_age)
        while True:
            yield functools.partial(cls, fn, mode, max_age=None)