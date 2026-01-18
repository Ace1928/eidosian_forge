import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def _get_ports(self, get_port):
    ports = []
    port_names = self.get_port_name_list()
    for name in port_names:
        if self.get_ofport(name) < 0:
            continue
        port = get_port(name)
        if port:
            ports.append(port)
    return ports