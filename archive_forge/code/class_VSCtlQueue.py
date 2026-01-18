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
class VSCtlQueue(object):

    def __init__(self, vsctl_qos_parent, ovsrec_queue):
        super(VSCtlQueue, self).__init__()
        self.qos = weakref.ref(vsctl_qos_parent)
        self.queue_cfg = ovsrec_queue