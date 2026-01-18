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
def _del_port(self, ctx, br_name=None, target=None, must_exist=False, with_iface=False):
    assert target is not None
    ctx.populate_cache()
    if not with_iface:
        vsctl_port = ctx.find_port(target, must_exist)
    else:
        vsctl_port = ctx.find_port(target, False)
        if not vsctl_port:
            vsctl_iface = ctx.find_iface(target, False)
            if vsctl_iface:
                vsctl_port = vsctl_iface.port()
            if must_exist and (not vsctl_port):
                vsctl_fatal('no port or interface named %s' % target)
    if not vsctl_port:
        return
    if not br_name:
        vsctl_bridge = ctx.find_bridge(br_name, True)
        if vsctl_port.bridge() != vsctl_bridge:
            if vsctl_port.bridge().parent == vsctl_bridge:
                vsctl_fatal('bridge %s does not have a port %s (although its parent bridge %s does)' % (br_name, target, vsctl_bridge.parent.name))
            else:
                vsctl_fatal('bridge %s does not have a port %s' % (br_name, target))
    ctx.del_port(vsctl_port)