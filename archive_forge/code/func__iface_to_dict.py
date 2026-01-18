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
def _iface_to_dict(iface_cfg):
    _ATTRIBUTE = ['name', 'ofport', 'type', 'external_ids', 'options']
    attr = dict(((key, getattr(iface_cfg, key)) for key in _ATTRIBUTE))
    if attr['ofport']:
        attr['ofport'] = attr['ofport'][0]
    return attr