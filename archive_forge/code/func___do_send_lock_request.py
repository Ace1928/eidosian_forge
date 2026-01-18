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
def __do_send_lock_request(self, method):
    self.__update_has_lock(False)
    self._lock_request_id = None
    if self._session.is_connected():
        msg = ovs.jsonrpc.Message.create_request(method, [self.lock_name])
        msg_id = msg.id
        self._session.send(msg)
    else:
        msg_id = None
    return msg_id