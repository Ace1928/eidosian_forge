import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
def _update_xmit_period(self):
    """
        Update transmission period of the BFD session.
        """
    if self._desired_min_tx_interval > self._remote_min_rx_interval:
        xmit_period = self._desired_min_tx_interval
    else:
        xmit_period = self._remote_min_rx_interval
    if self._detect_mult == 1:
        xmit_period *= random.randint(75, 90) / 100.0
    else:
        xmit_period *= random.randint(75, 100) / 100.0
    self._xmit_period = xmit_period / 1000000.0
    LOG.info('[BFD][%s][XMIT] Transmission period changed to %f', hex(self._local_discr), self._xmit_period)