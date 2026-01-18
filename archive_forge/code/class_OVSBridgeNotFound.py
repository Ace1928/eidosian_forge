import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
class OVSBridgeNotFound(os_ken_exc.OSKenException):
    message = 'no bridge for datapath_id %(datapath_id)s'