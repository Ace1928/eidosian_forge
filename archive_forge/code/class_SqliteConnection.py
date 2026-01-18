from contextlib import contextmanager
import sqlite3
from eventlet import sleep
from eventlet import timeout
from oslo_log import log as logging
from glance.i18n import _LE
class SqliteConnection(sqlite3.Connection):
    """
    SQLite DB Connection handler that plays well with eventlet,
    slightly modified from Swift's similar code.
    """

    def __init__(self, *args, **kwargs):
        self.timeout_seconds = kwargs.get('timeout', DEFAULT_SQL_CALL_TIMEOUT)
        kwargs['timeout'] = 0
        sqlite3.Connection.__init__(self, *args, **kwargs)

    def _timeout(self, call):
        with timeout.Timeout(self.timeout_seconds):
            while True:
                try:
                    return call()
                except sqlite3.OperationalError as e:
                    if 'locked' not in str(e):
                        raise
                sleep(0.05)

    def execute(self, *args, **kwargs):
        return self._timeout(lambda: sqlite3.Connection.execute(self, *args, **kwargs))

    def commit(self):
        return self._timeout(lambda: sqlite3.Connection.commit(self))