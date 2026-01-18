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
class ARPPacket(object):
    """
    ARPPacket class for parsing raw ARP packet, and generating ARP packet with
    Ethernet header.
    """

    class ARPUnknownFormat(OSKenException):
        message = '%(msg)s'

    @staticmethod
    def arp_packet(opcode, src_mac, src_ip, dst_mac, dst_ip):
        """
        Generate ARP packet with ethernet encapsulated.
        """
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet(dst_mac, src_mac, ETH_TYPE_ARP)
        pkt.add_protocol(eth_pkt)
        arp_pkt = arp.arp_ip(opcode, src_mac, src_ip, dst_mac, dst_ip)
        pkt.add_protocol(arp_pkt)
        pkt.serialize()
        return pkt.data

    @staticmethod
    def arp_parse(data):
        """
        Parse ARP packet, return ARP class from packet library.
        """
        pkt = packet.Packet(data)
        i = iter(pkt)
        eth_pkt = next(i)
        assert isinstance(eth_pkt, ethernet.ethernet)
        arp_pkt = next(i)
        if not isinstance(arp_pkt, arp.arp):
            raise ARPPacket.ARPUnknownFormat()
        if arp_pkt.opcode not in (ARP_REQUEST, ARP_REPLY):
            raise ARPPacket.ARPUnknownFormat(msg='unsupported opcode %d' % arp_pkt.opcode)
        if arp_pkt.proto != ETH_TYPE_IP:
            raise ARPPacket.ARPUnknownFormat(msg='unsupported arp ethtype 0x%04x' % arp_pkt.proto)
        return arp_pkt