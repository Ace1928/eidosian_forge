import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
class ColumnDefaultDict(dict):
    """A column dictionary with on-demand generated default values

    This object acts like the Row._data column dictionary, but without the
    necessity of populating column default values. These values are generated
    on-demand and therefore only use memory once they are accessed.
    """
    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table
        super().__init__()

    def __missing__(self, column):
        column = self._table.columns[column]
        return ovs.db.data.Datum.default(column.type)

    def keys(self):
        return self._table.columns.keys()

    def values(self):
        return iter((self[k] for k in self))

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()