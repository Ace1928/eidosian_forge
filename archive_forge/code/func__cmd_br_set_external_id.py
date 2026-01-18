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
def _cmd_br_set_external_id(self, ctx, command):
    br_name = command.args[0]
    key = command.args[1]
    if len(command.args) > 2:
        self._br_add_external_id(ctx, br_name, key, command.args[2])
    else:
        self._br_clear_external_id(ctx, br_name, key)