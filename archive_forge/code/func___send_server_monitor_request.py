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
def __send_server_monitor_request(self):
    self.state = self.IDL_S_SERVER_MONITOR_REQUESTED
    monitor_requests = {}
    table = self.server_tables[self._server_db_table]
    columns = [column for column in table.columns.keys()]
    for column in table.columns.values():
        if not hasattr(column, 'alert'):
            column.alert = True
    table.rows = custom_index.IndexedRows(table)
    table.need_table = False
    table.idl = self
    monitor_request = {'columns': columns}
    monitor_requests[table.name] = [monitor_request]
    msg = ovs.jsonrpc.Message.create_request('monitor', [self._server_db.name, str(self.server_monitor_uuid), monitor_requests])
    self._server_monitor_request_id = msg.id
    self.send_request(msg)