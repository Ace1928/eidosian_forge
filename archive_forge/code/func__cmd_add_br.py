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
def _cmd_add_br(self, ctx, command):
    br_name = command.args[0]
    parent_name = None
    vlan = 0
    if len(command.args) == 1:
        pass
    elif len(command.args) == 3:
        parent_name = command.args[1]
        vlan = int(command.args[2])
        if vlan < 0 or vlan > 4095:
            vsctl_fatal('vlan must be between 0 and 4095 %d' % vlan)
    else:
        vsctl_fatal('this command takes exactly 1 or 3 argument')
    ctx.add_bridge(br_name, parent_name, vlan)