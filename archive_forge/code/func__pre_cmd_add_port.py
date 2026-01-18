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
def _pre_cmd_add_port(self, ctx, command):
    self._pre_get_info(ctx, command)
    columns = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting)[0] for setting in command.args[2:]]
    self._pre_add_port(ctx, columns)