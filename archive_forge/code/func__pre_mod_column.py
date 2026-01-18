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
def _pre_mod_column(self, ovsrec_row, column, value_json):
    if column not in ovsrec_row._table.columns:
        vsctl_fatal('%s does not contain a column whose name matches "%s"' % (ovsrec_row._table.name, column))
    column_schema = ovsrec_row._table.columns[column]
    datum = ovs.db.data.Datum.from_json(column_schema.type, value_json, self.symtab)
    return datum.to_python(ovs.db.idl._uuid_to_row)