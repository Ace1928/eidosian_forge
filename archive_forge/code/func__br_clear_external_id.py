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
def _br_clear_external_id(self, ctx, br_name, key):
    table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
    column = vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS
    vsctl_table = self._get_table(table_name)
    ovsrec_row = ctx.must_get_row(vsctl_table, br_name)
    values = getattr(ovsrec_row, column, {})
    values.pop(key, None)
    setattr(ovsrec_row, column, values)
    ctx.invalidate_cache()