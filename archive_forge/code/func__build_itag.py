import logging
import struct
import unittest
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import vlan
from os_ken.lib.packet import pbb
def _build_itag(self):
    b_src_mac = '00:07:0d:af:f4:54'
    b_dst_mac = '00:00:00:00:00:00'
    b_ethertype = ether.ETH_TYPE_8021AD
    e1 = ethernet.ethernet(b_dst_mac, b_src_mac, b_ethertype)
    b_pcp = 0
    b_cfi = 0
    b_vid = 32
    b_ethertype = ether.ETH_TYPE_8021Q
    bt = vlan.svlan(b_pcp, b_cfi, b_vid, b_ethertype)
    c_src_mac = '11:11:11:11:11:11'
    c_dst_mac = 'aa:aa:aa:aa:aa:aa'
    c_ethertype = ether.ETH_TYPE_8021AD
    e2 = ethernet.ethernet(c_dst_mac, c_src_mac, c_ethertype)
    s_pcp = 0
    s_cfi = 0
    s_vid = 32
    s_ethertype = ether.ETH_TYPE_8021Q
    st = vlan.svlan(s_pcp, s_cfi, s_vid, s_ethertype)
    c_pcp = 0
    c_cfi = 0
    c_vid = 32
    c_ethertype = ether.ETH_TYPE_IP
    ct = vlan.vlan(c_pcp, c_cfi, c_vid, c_ethertype)
    version = 4
    header_length = 20
    tos = 0
    total_length = 24
    identification = 35421
    flags = 0
    offset = 1480
    ttl = 64
    proto = inet.IPPROTO_ICMP
    csum = 42994
    src = '131.151.32.21'
    dst = '131.151.32.129'
    option = b'TEST'
    ip = ipv4.ipv4(version, header_length, tos, total_length, identification, flags, offset, ttl, proto, csum, src, dst, option)
    p = packet.Packet()
    p.add_protocol(e1)
    p.add_protocol(bt)
    p.add_protocol(self.it)
    p.add_protocol(e2)
    p.add_protocol(st)
    p.add_protocol(ct)
    p.add_protocol(ip)
    p.serialize()
    return p