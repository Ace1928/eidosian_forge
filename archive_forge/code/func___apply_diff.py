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
def __apply_diff(self, table, row, row_diff):
    old_row = {}
    for column_name, datum_diff_json in row_diff.items():
        column = table.columns.get(column_name)
        if not column:
            vlog.warn('unknown column %s updating table %s' % (column_name, table.name))
            continue
        try:
            datum_diff = data.Datum.from_json(column.type, datum_diff_json)
        except error.Error as e:
            vlog.warn('error parsing column %s in table %s: %s' % (column_name, table.name, e))
            continue
        old_row[column_name] = row._data[column_name].copy()
        datum = row._data[column_name].diff(datum_diff)
        if datum != row._data[column_name]:
            row._data[column_name] = datum
    return old_row