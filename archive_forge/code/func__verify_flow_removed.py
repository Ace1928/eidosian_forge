import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _verify_flow_removed(self, dp, msg):
    params = self._verify['params']
    in_port = self._verify['in_port']
    timeout = self._verify['timeout']
    if timeout:
        duration_nsec = msg.duration_sec * 10 ** 9 + msg.duration_nsec
        timeout_nsec = timeout * 10 ** 9
        l = (timeout - 0.5) * 10 ** 9
        h = (timeout + 1.5) * 10 ** 9
        if not l < duration_nsec < h:
            return 'bad duration time. set=%s(nsec), duration=%s(nsec)' % (timeout_nsec, duration_nsec)
    for name, val in params.items():
        r_val = getattr(msg, name)
        if val != r_val:
            return '%s is mismatched. verify=%s, reply=%s' % (name, val, r_val)
    for f in msg.match.fields:
        if f.header == ofproto_v1_2.OXM_OF_IN_PORT:
            if f.value != in_port:
                return 'in_port is mismatched. verify=%s, reply=%s' % (in_port, f.value)
    return True