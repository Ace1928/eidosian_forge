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
def _uuid_to_row(atom, base):
    if base.ref_table:
        value = base.ref_table.rows.get(atom)
    else:
        value = atom
    if isinstance(value, idl.Row):
        value = str(value.uuid)
    return value