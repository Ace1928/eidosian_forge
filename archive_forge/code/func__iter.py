import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _iter(self, db):
    c = db.cursor(cursorclass=MySQLdb.cursors.SSCursor)
    c.execute('SELECT digest FROM %s' % self.table_name)
    while True:
        row = c.fetchone()
        if not row:
            break
        yield row[0]
    c.close()