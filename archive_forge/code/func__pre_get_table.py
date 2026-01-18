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
def _pre_get_table(self, _ctx, table_name):
    vsctl_table = self._get_table(table_name)
    schema_helper = self.schema_helper
    schema_helper.register_table(vsctl_table.table_name)
    for row_id in vsctl_table.row_ids:
        if row_id.table:
            schema_helper.register_table(row_id.table)
        if row_id.name_column:
            schema_helper.register_columns(row_id.table, [row_id.name_column])
        if row_id.uuid_column:
            schema_helper.register_columns(row_id.table, [row_id.uuid_column])
    return vsctl_table