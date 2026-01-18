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
class _VSCtlTable(object):

    def __init__(self, table_name, vsctl_row_id_list):
        super(_VSCtlTable, self).__init__()
        self.table_name = table_name
        self.row_ids = vsctl_row_id_list