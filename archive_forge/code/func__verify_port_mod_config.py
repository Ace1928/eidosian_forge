import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _verify_port_mod_config(self, dp, msg):
    port_no = self._verify[0]
    config = self._verify[1]
    port = msg.ports[port_no]
    if config != port.config:
        return 'config is mismatched. verify=%s, stats=%s' % (bin(config), bin(port.config))
    return True