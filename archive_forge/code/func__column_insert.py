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
def _column_insert(ovsrec_row, column, ovsrec_add):
    value = getattr(ovsrec_row, column)
    value.append(ovsrec_add)
    VSCtlContext._column_set(ovsrec_row, column, value)