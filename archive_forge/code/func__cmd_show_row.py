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
@staticmethod
def _cmd_show_row(ctx, row, level):
    _INDENT_SIZE = 4
    show = VSCtl._cmd_show_find_table_by_row(row)
    output = ''
    output += ' ' * level * _INDENT_SIZE
    if show and show.name_column:
        output += '%s ' % show.table
        datum = getattr(row, show.name_column)
        output += datum
    else:
        output += str(row.uuid)
    output += '\n'
    if not show or show.recurse:
        return
    show.recurse = True
    for column in show.columns:
        datum = row._data[column]
        key = datum.type.key
        if key.type == ovs.db.types.UuidType and key.ref_table_name:
            ref_show = VSCtl._cmd_show_find_table_by_name(key.ref_table_name)
            if ref_show:
                for atom in datum.values:
                    ref_row = ctx.idl.tables[ref_show.table].rows.get(atom.value)
                    if ref_row:
                        VSCtl._cmd_show_row(ctx, ref_row, level + 1)
                continue
        if not datum.is_default():
            output += ' ' * (level + 1) * _INDENT_SIZE
            output += '%s: %s\n' % (column, datum)
    show.recurse = False
    return output