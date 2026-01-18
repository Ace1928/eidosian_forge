import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _add_flow_for_flow_mod_tests(self, dp):
    a1 = dp.ofproto_parser.OFPActionOutput(1, 1500)
    a2 = dp.ofproto_parser.OFPActionOutput(2, 1500)
    tables = {0: [65535, 10, b'\xee' * 6, a1], 1: [65280, 10, b'\xee' * 6, a2], 2: [61440, 100, b'\xee' * 6, a1], 3: [0, 10, b'\xff' * 6, a1]}
    self._verify = tables
    for table_id, val in tables.items():
        match = dp.ofproto_parser.OFPMatch()
        match.set_dl_dst(val[2])
        self.mod_flow(dp, match=match, actions=[val[3]], table_id=table_id, cookie=val[0], priority=val[1])
    dp.send_barrier()