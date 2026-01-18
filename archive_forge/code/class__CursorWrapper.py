import os
from . import BioSeq
from . import Loader
from . import DBUtils
class _CursorWrapper:
    """A wrapper for mysql.connector resolving bytestring representations."""

    def __init__(self, real_cursor):
        self.real_cursor = real_cursor

    def execute(self, operation, params=None, multi=False):
        """Execute a sql statement."""
        self.real_cursor.execute(operation, params, multi)

    def executemany(self, operation, params):
        """Execute many sql statements."""
        self.real_cursor.executemany(operation, params)

    def _convert_tuple(self, tuple_):
        """Decode any bytestrings present in the row (PRIVATE)."""
        tuple_list = list(tuple_)
        for i, elem in enumerate(tuple_list):
            if isinstance(elem, bytes):
                tuple_list[i] = elem.decode('utf-8')
        return tuple(tuple_list)

    def _convert_list(self, lst):
        ret_lst = []
        for tuple_ in lst:
            new_tuple = self._convert_tuple(tuple_)
            ret_lst.append(new_tuple)
        return ret_lst

    def fetchall(self):
        rv = self.real_cursor.fetchall()
        return self._convert_list(rv)

    def fetchone(self):
        tuple_ = self.real_cursor.fetchone()
        return self._convert_tuple(tuple_)