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
class VSCtlIface(object):

    def __init__(self, vsctl_port_parent, ovsrec_iface):
        super(VSCtlIface, self).__init__()
        self.port = weakref.ref(vsctl_port_parent)
        self.iface_cfg = ovsrec_iface