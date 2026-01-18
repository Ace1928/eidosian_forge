import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _verify_packet_in(self, dp, msg):
    for name, val in self._verify.items():
        if name == 'in_port':
            for f in msg.match.fields:
                if f.header == ofproto_v1_2.OXM_OF_IN_PORT:
                    r_val = f.value
        else:
            r_val = getattr(msg, name)
        if val != r_val:
            return '%s is mismatched. verify=%s, reply=%s' % (name, val, r_val)
    return True