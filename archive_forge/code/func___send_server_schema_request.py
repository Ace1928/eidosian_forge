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
def __send_server_schema_request(self):
    self.state = self.IDL_S_SERVER_SCHEMA_REQUESTED
    msg = ovs.jsonrpc.Message.create_request('get_schema', [self._server_db_name, str(self.uuid)])
    self._server_schema_request_id = msg.id
    self.send_request(msg)