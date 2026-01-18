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
def __process_insert_reply(self, insert, ops):
    if insert.op_index >= len(ops):
        vlog.warn('reply does not contain enough operations for insert (has %d, needs %d)' % (len(ops), insert.op_index))
        return False
    reply = ops[insert.op_index]
    json_uuid = reply.get('uuid')
    if not Transaction.__check_json_type(json_uuid, (tuple, list), '"insert" reply "uuid"'):
        return False
    try:
        uuid_ = ovs.ovsuuid.from_json(json_uuid)
    except error.Error:
        vlog.warn('"insert" reply "uuid" is not a JSON UUID')
        return False
    insert.real = uuid_
    return True