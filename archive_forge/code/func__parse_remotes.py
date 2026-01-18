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
def _parse_remotes(self, remote):
    remotes = []
    for r in remote.split(','):
        if remotes and r.find(':') == -1:
            remotes[-1] += ',' + r
        else:
            remotes.append(r)
    return remotes