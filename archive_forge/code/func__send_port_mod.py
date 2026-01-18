import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _send_port_mod(self, dp, config, mask):
    p = self.get_port(dp)
    if not p:
        err = 'need attached port to switch.'
        self.results[self.current] = err
        self.start_next_test(dp)
        return
    self._verify = [p.port_no, config & mask]
    m = dp.ofproto_parser.OFPPortMod(dp, p.port_no, p.hw_addr, config, mask, 0)
    dp.send_msg(m)
    dp.send_barrier()
    time.sleep(1)
    m = dp.ofproto_parser.OFPFeaturesRequest(dp)
    dp.send_msg(m)