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
def __send_db_change_aware(self):
    msg = ovs.jsonrpc.Message.create_request('set_db_change_aware', [True])
    self._db_change_aware_request_id = msg.id
    self._session.send(msg)