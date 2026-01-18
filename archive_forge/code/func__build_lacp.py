import copy
import logging
from struct import pack, unpack_from
import unittest
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib import addrconv
from os_ken.lib.packet.slow import slow, lacp
from os_ken.lib.packet.slow import SLOW_PROTOCOL_MULTICAST
from os_ken.lib.packet.slow import SLOW_SUBTYPE_LACP
from os_ken.lib.packet.slow import SLOW_SUBTYPE_MARKER
def _build_lacp(self):
    ethertype = ether.ETH_TYPE_SLOW
    dst = SLOW_PROTOCOL_MULTICAST
    e = ethernet(dst, self.actor_system, ethertype)
    p = Packet()
    p.add_protocol(e)
    p.add_protocol(self.l)
    p.serialize()
    return p