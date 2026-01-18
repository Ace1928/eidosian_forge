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
@staticmethod
def __check_json_type(json, types, name):
    if not json:
        vlog.warn('%s is missing' % name)
        return False
    elif not isinstance(json, tuple(types)):
        vlog.warn('%s has unexpected type %s' % (name, type(json)))
        return False
    else:
        return True