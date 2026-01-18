import collections
import functools
import operator
from ovs.db import data
class MultiColumnIndex(object):

    def __init__(self, name):
        self.name = name
        self.columns = []
        self.clear()

    def __repr__(self):
        return '{}(name={})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return repr(self) + ' columns={} values={}'.format(self.columns, [str(v) for v in self.values])

    def add_column(self, column, direction=OVSDB_INDEX_ASC, key=None):
        self.columns.append(ColumnIndex(column, direction, key or operator.attrgetter(column)))

    def add_columns(self, *columns):
        self.columns.extend((ColumnIndex(col, OVSDB_INDEX_ASC, operator.attrgetter(col)) for col in columns))

    def _cmp(self, a, b):
        for col, direction, key in self.columns:
            aval, bval = (key(a), key(b))
            if aval == bval:
                continue
            result = (aval > bval) - (aval < bval)
            return result if direction == OVSDB_INDEX_ASC else -result
        return 0

    def index_entry_from_row(self, row):
        return row._table.rows.IndexEntry(uuid=row.uuid, **{c.column: getattr(row, c.column) for c in self.columns})

    def add(self, row):
        if not all((hasattr(row, col.column) for col in self.columns)):
            return
        self.values.add(self.index_entry_from_row(row))

    def remove(self, row):
        self.values.remove(self.index_entry_from_row(row))

    def clear(self):
        self.values = sortedcontainers.SortedListWithKey(key=functools.cmp_to_key(self._cmp))

    def irange(self, start, end):
        return iter((r._table.rows[r.uuid] for r in self.values.irange(start, end)))

    def __iter__(self):
        return iter((r._table.rows[r.uuid] for r in self.values))