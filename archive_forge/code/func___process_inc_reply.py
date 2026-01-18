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
def __process_inc_reply(self, ops):
    if self._inc_index + 2 > len(ops):
        vlog.warn('reply does not contain enough operations for increment (has %d, needs %d)' % (len(ops), self._inc_index + 2))
    mutate = ops[self._inc_index]
    count = mutate.get('count')
    if not Transaction.__check_json_type(count, (int,), '"mutate" reply "count"'):
        return False
    if count != 1:
        vlog.warn('"mutate" reply "count" is %d instead of 1' % count)
        return False
    select = ops[self._inc_index + 1]
    rows = select.get('rows')
    if not Transaction.__check_json_type(rows, (list, tuple), '"select" reply "rows"'):
        return False
    if len(rows) != 1:
        vlog.warn('"select" reply "rows" has %d elements instead of 1' % len(rows))
        return False
    row = rows[0]
    if not Transaction.__check_json_type(row, (dict,), '"select" reply row'):
        return False
    column = row.get(self._inc_column)
    if not Transaction.__check_json_type(column, (int,), '"select" reply inc column'):
        return False
    self._inc_new_value = column
    return True