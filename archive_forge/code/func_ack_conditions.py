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
def ack_conditions(self):
    """Mark all requested table conditions as acked"""
    for table in self.tables.values():
        table.condition_state.ack()