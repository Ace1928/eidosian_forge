import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
class ProcessMySQLDBHandle(MySQLDBHandle):

    def __init__(self, fn, mode, max_age=None):
        MySQLDBHandle.__init__(self, fn, mode, max_age=max_age)

    def reconnect(self):
        pass

    def __del__(self):
        pass

    def _safe_call(self, name, method, args):
        db = None
        try:
            db = self._get_new_connection()
            return method(*args, db=db)
        except (MySQLdb.Error, AttributeError) as ex:
            self.log.error('%s failed: %s', name, ex)
            raise DatabaseError('Database temporarily unavailable.')
        finally:
            if db is not None:
                db.close()