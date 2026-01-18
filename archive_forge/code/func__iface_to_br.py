import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def _iface_to_br(self, ctx, iface_name):
    ctx.populate_cache()
    iface = ctx.find_iface(iface_name, True)
    port = iface.port()
    if port is None:
        vsctl_fatal('Port associated to iface "%s" does not exist' % iface_name)
    bridge = port.bridge()
    if bridge is None:
        vsctl_fatal('Bridge associated to iface "%s" does not exist' % iface_name)
    return bridge.name