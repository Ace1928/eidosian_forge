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
def __add_default(self, table, row_update):
    for column in table.columns.values():
        if column.name not in row_update:
            if table.name not in self.readonly or (table.name in self.readonly and column.name not in self.readonly[table.name]):
                if column.type.n_min != 0 and (not column.type.is_map()):
                    row_update[column.name] = self.__column_name(column)