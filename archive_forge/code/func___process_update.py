import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def __process_update(self, table, uuid, old, new):
    old_row = table.rows.get(uuid)
    if old_row is not None:
        old_row = model.Row(dictify(old_row))
        old_row['_uuid'] = uuid
    changed = idl.Idl.__process_update(self, table, uuid, old, new)
    if changed:
        if not new:
            ev = (event.EventRowDelete, (table.name, old_row))
        elif not old:
            new_row = model.Row(dictify(table.rows.get(uuid)))
            new_row['_uuid'] = uuid
            ev = (event.EventRowInsert, (table.name, new_row))
        else:
            new_row = model.Row(dictify(table.rows.get(uuid)))
            new_row['_uuid'] = uuid
            ev = (event.EventRowUpdate, (table.name, old_row, new_row))
        self._events.append(ev)
    return changed