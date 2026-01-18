import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def _state_machine(self):
    """ Port state machine.
             Change next status when timer is exceeded
             or _change_status() method is called."""
    role_str = {ROOT_PORT: 'ROOT_PORT          ', DESIGNATED_PORT: 'DESIGNATED_PORT    ', NON_DESIGNATED_PORT: 'NON_DESIGNATED_PORT'}
    state_str = {PORT_STATE_DISABLE: 'DISABLE', PORT_STATE_BLOCK: 'BLOCK', PORT_STATE_LISTEN: 'LISTEN', PORT_STATE_LEARN: 'LEARN', PORT_STATE_FORWARD: 'FORWARD'}
    if self.state is PORT_STATE_DISABLE:
        self.ofctl.set_port_status(self.ofport, self.state)
    while True:
        self.logger.info('[port=%d] %s / %s', self.ofport.port_no, role_str[self.role], state_str[self.state], extra=self.dpid_str)
        self.state_event = hub.Event()
        timer = self._get_timer()
        if timer:
            timeout = hub.Timeout(timer)
            try:
                self.state_event.wait()
            except hub.Timeout as t:
                if t is not timeout:
                    err_msg = 'Internal error. Not my timeout.'
                    raise OSKenException(msg=err_msg)
                new_state = self._get_next_state()
                self._change_status(new_state, thread_switch=False)
            finally:
                timeout.cancel()
        else:
            self.state_event.wait()
        self.state_event = None