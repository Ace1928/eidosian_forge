import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def _get_bridge_name(self):
    """ get Bridge name of a given 'datapath_id' """
    command = ovs_vsctl.VSCtlCommand('find', ('Bridge', 'datapath_id=%s' % dpid_lib.dpid_to_str(self.datapath_id)))
    self.run_command([command])
    if not isinstance(command.result, list) or len(command.result) != 1:
        raise OVSBridgeNotFound(datapath_id=dpid_lib.dpid_to_str(self.datapath_id))
    return command.result[0].name