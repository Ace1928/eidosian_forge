import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
class ClsRule(ofproto_parser.StringifyMixin):
    """describe a matching rule for OF 1.0 OFPMatch (and NX).
    """

    def __init__(self, **kwargs):
        self.wc = FlowWildcards()
        self.flow = Flow()
        for key, value in kwargs.items():
            if key[:3] == 'reg':
                register = int(key[3:] or -1)
                self.set_reg(register, value)
                continue
            setter = getattr(self, 'set_' + key, None)
            if not setter:
                LOG.error('Invalid kwarg specified to ClsRule (%s)', key)
                continue
            if not isinstance(value, (tuple, list)):
                value = (value,)
            setter(*value)

    def set_in_port(self, port):
        self.wc.wildcards &= ~FWW_IN_PORT
        self.flow.in_port = port

    def set_dl_vlan(self, dl_vlan):
        self.wc.wildcards &= ~ofproto_v1_0.OFPFW_DL_VLAN
        self.flow.dl_vlan = dl_vlan

    def set_dl_vlan_pcp(self, dl_vlan_pcp):
        self.wc.wildcards &= ~ofproto_v1_0.OFPFW_DL_VLAN_PCP
        self.flow.dl_vlan_pcp = dl_vlan_pcp

    def set_dl_dst(self, dl_dst):
        self.flow.dl_dst = dl_dst

    def set_dl_dst_masked(self, dl_dst, mask):
        self.wc.dl_dst_mask = mask
        self.flow.dl_dst = mac.haddr_bitand(dl_dst, mask)

    def set_dl_src(self, dl_src):
        self.flow.dl_src = dl_src

    def set_dl_src_masked(self, dl_src, mask):
        self.wc.dl_src_mask = mask
        self.flow.dl_src = mac.haddr_bitand(dl_src, mask)

    def set_dl_type(self, dl_type):
        self.wc.wildcards &= ~FWW_DL_TYPE
        self.flow.dl_type = dl_type

    def set_dl_tci(self, tci):
        self.set_dl_tci_masked(tci, UINT16_MAX)

    def set_dl_tci_masked(self, tci, mask):
        self.wc.vlan_tci_mask = mask
        self.flow.vlan_tci = tci

    def set_tp_src(self, tp_src):
        self.set_tp_src_masked(tp_src, UINT16_MAX)

    def set_tp_src_masked(self, tp_src, mask):
        self.wc.tp_src_mask = mask
        self.flow.tp_src = tp_src & mask

    def set_tp_dst(self, tp_dst):
        self.set_tp_dst_masked(tp_dst, UINT16_MAX)

    def set_tp_dst_masked(self, tp_dst, mask):
        self.wc.tp_dst_mask = mask
        self.flow.tp_dst = tp_dst & mask

    def set_nw_proto(self, nw_proto):
        self.wc.wildcards &= ~FWW_NW_PROTO
        self.flow.nw_proto = nw_proto

    def set_nw_src(self, nw_src):
        self.set_nw_src_masked(nw_src, UINT32_MAX)

    def set_nw_src_masked(self, nw_src, mask):
        self.flow.nw_src = nw_src
        self.wc.nw_src_mask = mask

    def set_nw_dst(self, nw_dst):
        self.set_nw_dst_masked(nw_dst, UINT32_MAX)

    def set_nw_dst_masked(self, nw_dst, mask):
        self.flow.nw_dst = nw_dst
        self.wc.nw_dst_mask = mask

    def set_nw_dscp(self, nw_dscp):
        self.wc.wildcards &= ~FWW_NW_DSCP
        self.flow.nw_tos &= ~IP_DSCP_MASK
        self.flow.nw_tos |= nw_dscp & IP_DSCP_MASK

    def set_icmp_type(self, icmp_type):
        self.set_tp_src(icmp_type)

    def set_icmp_code(self, icmp_code):
        self.set_tp_dst(icmp_code)

    def set_tun_id(self, tun_id):
        self.set_tun_id_masked(tun_id, UINT64_MAX)

    def set_tun_id_masked(self, tun_id, mask):
        self.wc.tun_id_mask = mask
        self.flow.tun_id = tun_id & mask

    def set_nw_ecn(self, nw_ecn):
        self.wc.wildcards &= ~FWW_NW_ECN
        self.flow.nw_tos &= ~IP_ECN_MASK
        self.flow.nw_tos |= nw_ecn & IP_ECN_MASK

    def set_nw_ttl(self, nw_ttl):
        self.wc.wildcards &= ~FWW_NW_TTL
        self.flow.nw_ttl = nw_ttl

    def set_nw_frag(self, nw_frag):
        self.wc.nw_frag_mask |= FLOW_NW_FRAG_MASK
        self.flow.nw_frag = nw_frag

    def set_nw_frag_masked(self, nw_frag, mask):
        self.wc.nw_frag_mask = mask
        self.flow.nw_frag = nw_frag & mask

    def set_arp_spa(self, spa):
        self.set_arp_spa_masked(spa, UINT32_MAX)

    def set_arp_spa_masked(self, spa, mask):
        self.flow.arp_spa = spa
        self.wc.arp_spa_mask = mask

    def set_arp_tpa(self, tpa):
        self.set_arp_tpa_masked(tpa, UINT32_MAX)

    def set_arp_tpa_masked(self, tpa, mask):
        self.flow.arp_tpa = tpa
        self.wc.arp_tpa_mask = mask

    def set_arp_sha(self, sha):
        self.wc.wildcards &= ~FWW_ARP_SHA
        self.flow.arp_sha = sha

    def set_arp_tha(self, tha):
        self.wc.wildcards &= ~FWW_ARP_THA
        self.flow.arp_tha = tha

    def set_icmpv6_type(self, icmp_type):
        self.set_tp_src(icmp_type)

    def set_icmpv6_code(self, icmp_code):
        self.set_tp_dst(icmp_code)

    def set_ipv6_label(self, label):
        self.wc.wildcards &= ~FWW_IPV6_LABEL
        self.flow.ipv6_label = label

    def set_ipv6_src_masked(self, src, mask):
        self.wc.ipv6_src_mask = mask
        self.flow.ipv6_src = [x & y for x, y in zip(src, mask)]

    def set_ipv6_src(self, src):
        self.flow.ipv6_src = src

    def set_ipv6_dst_masked(self, dst, mask):
        self.wc.ipv6_dst_mask = mask
        self.flow.ipv6_dst = [x & y for x, y in zip(dst, mask)]

    def set_ipv6_dst(self, dst):
        self.flow.ipv6_dst = dst

    def set_nd_target_masked(self, target, mask):
        self.wc.nd_target_mask = mask
        self.flow.nd_target = [x & y for x, y in zip(target, mask)]

    def set_nd_target(self, target):
        self.flow.nd_target = target

    def set_reg(self, reg_idx, value):
        self.set_reg_masked(reg_idx, value, 0)

    def set_reg_masked(self, reg_idx, value, mask):
        self.wc.regs_mask[reg_idx] = mask
        self.flow.regs[reg_idx] = value
        self.wc.regs_bits |= 1 << reg_idx

    def set_pkt_mark_masked(self, pkt_mark, mask):
        self.flow.pkt_mark = pkt_mark
        self.wc.pkt_mark_mask = mask

    def set_tcp_flags(self, tcp_flags, mask):
        self.flow.tcp_flags = tcp_flags
        self.wc.tcp_flags_mask = mask

    def flow_format(self):
        if self.wc.tun_id_mask != 0:
            return ofproto_v1_0.NXFF_NXM
        if self.wc.dl_dst_mask:
            return ofproto_v1_0.NXFF_NXM
        if self.wc.dl_src_mask:
            return ofproto_v1_0.NXFF_NXM
        if not self.wc.wildcards & FWW_NW_ECN:
            return ofproto_v1_0.NXFF_NXM
        if self.wc.regs_bits > 0:
            return ofproto_v1_0.NXFF_NXM
        if self.flow.tcp_flags > 0:
            return ofproto_v1_0.NXFF_NXM
        return ofproto_v1_0.NXFF_OPENFLOW10

    def match_tuple(self):
        """return a tuple which can be used as *args for
        ofproto_v1_0_parser.OFPMatch.__init__().
        see Datapath.send_flow_mod.
        """
        assert self.flow_format() == ofproto_v1_0.NXFF_OPENFLOW10
        wildcards = ofproto_v1_0.OFPFW_ALL
        if not self.wc.wildcards & FWW_IN_PORT:
            wildcards &= ~ofproto_v1_0.OFPFW_IN_PORT
        if self.flow.dl_src != mac.DONTCARE:
            wildcards &= ~ofproto_v1_0.OFPFW_DL_SRC
        if self.flow.dl_dst != mac.DONTCARE:
            wildcards &= ~ofproto_v1_0.OFPFW_DL_DST
        if not self.wc.wildcards & FWW_DL_TYPE:
            wildcards &= ~ofproto_v1_0.OFPFW_DL_TYPE
        if self.flow.dl_vlan != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_DL_VLAN
        if self.flow.dl_vlan_pcp != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_DL_VLAN_PCP
        if self.flow.nw_tos != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_NW_TOS
        if self.flow.nw_proto != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_NW_PROTO
        if self.wc.nw_src_mask != 0 and '01' not in bin(self.wc.nw_src_mask):
            wildcards &= ~ofproto_v1_0.OFPFW_NW_SRC_MASK
            maskbits = bin(self.wc.nw_src_mask).count('0') - 1
            wildcards |= maskbits << ofproto_v1_0.OFPFW_NW_SRC_SHIFT
        if self.wc.nw_dst_mask != 0 and '01' not in bin(self.wc.nw_dst_mask):
            wildcards &= ~ofproto_v1_0.OFPFW_NW_DST_MASK
            maskbits = bin(self.wc.nw_dst_mask).count('0') - 1
            wildcards |= maskbits << ofproto_v1_0.OFPFW_NW_DST_SHIFT
        if self.flow.tp_src != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_TP_SRC
        if self.flow.tp_dst != 0:
            wildcards &= ~ofproto_v1_0.OFPFW_TP_DST
        return (wildcards, self.flow.in_port, self.flow.dl_src, self.flow.dl_dst, self.flow.dl_vlan, self.flow.dl_vlan_pcp, self.flow.dl_type, self.flow.nw_tos & IP_DSCP_MASK, self.flow.nw_proto, self.flow.nw_src, self.flow.nw_dst, self.flow.tp_src, self.flow.tp_dst)