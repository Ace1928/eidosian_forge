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
def __disassemble(self):
    self.idl.txn = None
    for row in self._txn_rows.values():
        if row._changes is None:
            row.__dict__['_changes'] = {}
            row.__dict__['_mutations'] = {}
            row._table.rows[row.uuid] = row
        elif row._data is None:
            del row._table.rows[row.uuid]
        row.__dict__['_changes'] = {}
        row.__dict__['_mutations'] = {}
        row.__dict__['_prereqs'] = {}
    self._txn_rows = {}