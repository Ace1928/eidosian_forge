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
def _cmd_del_port(self, ctx, command):
    must_exist = command.has_option('--must-exist')
    with_iface = command.has_option('--with-iface')
    target = command.args[-1]
    br_name = command.args[0] if len(command.args) == 2 else None
    self._del_port(ctx, br_name, target, must_exist, with_iface)