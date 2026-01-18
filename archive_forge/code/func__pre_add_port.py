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
def _pre_add_port(self, _ctx, columns):
    schema_helper = self.schema_helper
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, [vswitch_idl.OVSREC_PORT_COL_NAME, vswitch_idl.OVSREC_PORT_COL_BOND_FAKE_IFACE])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, columns)