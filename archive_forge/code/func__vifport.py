import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def _vifport(self, name, external_ids):
    ofport = self.get_ofport(name)
    return VifPort(name, ofport, external_ids['iface-id'], external_ids['attached-mac'], self)