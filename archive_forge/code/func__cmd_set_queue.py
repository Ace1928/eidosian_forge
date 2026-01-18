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
def _cmd_set_queue(self, ctx, command):
    ctx.populate_cache()
    port_name = command.args[0]
    queues = command.args[1]
    vsctl_port = ctx.find_port(port_name, True)
    vsctl_qos = vsctl_port.qos
    queue_id = 0
    results = []
    for queue in queues:
        max_rate = queue.get('max-rate', None)
        min_rate = queue.get('min-rate', None)
        ovsrec_queue = ctx.set_queue(vsctl_qos, max_rate, min_rate, queue_id)
        results.append(ovsrec_queue)
        queue_id += 1
    command.result = results