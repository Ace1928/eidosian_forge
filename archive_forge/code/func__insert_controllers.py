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
def _insert_controllers(self, controller_names):
    ovsrec_controllers = []
    for name in controller_names:
        ovsrec_controller = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_CONTROLLER])
        ovsrec_controller.target = name
        ovsrec_controllers.append(ovsrec_controller)
    return ovsrec_controllers