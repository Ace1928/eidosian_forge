import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YChildFuncStats(YStatsIndexable):

    def sort(self, sort_type, sort_order='desc'):
        sort_type = _validate_sorttype(sort_type, SORT_TYPES_CHILDFUNCSTATS)
        sort_order = _validate_sortorder(sort_order)
        return super().sort(SORT_TYPES_CHILDFUNCSTATS[sort_type], SORT_ORDERS[sort_order])

    def print_all(self, out=sys.stdout, columns={0: ('name', 36), 1: ('ncall', 5), 2: ('tsub', 8), 3: ('ttot', 8), 4: ('tavg', 8)}):
        """
        Prints all of the child function profiler results to a given file. (stdout by default)
        """
        if self.empty() or len(columns) == 0:
            return
        for _, col in columns.items():
            _validate_columns(col[0], COLUMNS_FUNCSTATS)
        out.write(LINESEP)
        self._print_header(out, columns)
        for stat in self:
            stat._print(out, columns)

    def strip_dirs(self):
        for stat in self:
            stat.strip_dirs()
        return self