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
def __parse_lock_notify(self, params, new_has_lock):
    if self.lock_name is not None and isinstance(params, (list, tuple)) and params and (params[0] == self.lock_name):
        self.__update_has_lock(new_has_lock)
        if not new_has_lock:
            self.is_lock_contended = True