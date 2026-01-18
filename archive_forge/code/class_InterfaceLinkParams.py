import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
class InterfaceLinkParams(stringify.StringifyMixin):
    """
    Interface Link Parameters class for if_link_params structure.
    """
    _HEADER_FMT = '!IIffI'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _REPEATED_FMT = '!f'
    REPEATED_SIZE = struct.calcsize(_REPEATED_FMT)
    _FOOTER_FMT = '!II4sIIIIffff'
    FOOTER_SIZE = struct.calcsize(_FOOTER_FMT)

    def __init__(self, lp_status, te_metric, max_bw, max_reserved_bw, unreserved_bw, admin_group, remote_as, remote_ip, average_delay, min_delay, max_delay, delay_var, pkt_loss, residual_bw, average_bw, utilized_bw):
        super(InterfaceLinkParams, self).__init__()
        self.lp_status = lp_status
        self.te_metric = te_metric
        self.max_bw = max_bw
        self.max_reserved_bw = max_reserved_bw
        assert isinstance(unreserved_bw, (list, tuple))
        assert len(unreserved_bw) == MAX_CLASS_TYPE
        self.unreserved_bw = unreserved_bw
        self.admin_group = admin_group
        self.remote_as = remote_as
        assert ip.valid_ipv4(remote_ip)
        self.remote_ip = remote_ip
        self.average_delay = average_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_var = delay_var
        self.pkt_loss = pkt_loss
        self.residual_bw = residual_bw
        self.average_bw = average_bw
        self.utilized_bw = utilized_bw

    @classmethod
    def parse(cls, buf):
        lp_status, te_metric, max_bw, max_reserved_bw, bw_cls_num = struct.unpack_from(cls._HEADER_FMT, buf)
        if MAX_CLASS_TYPE < bw_cls_num:
            bw_cls_num = MAX_CLASS_TYPE
        offset = cls.HEADER_SIZE
        unreserved_bw = []
        for _ in range(bw_cls_num):
            u_bw, = struct.unpack_from(cls._REPEATED_FMT, buf, offset)
            unreserved_bw.append(u_bw)
            offset += cls.REPEATED_SIZE
        admin_group, remote_as, remote_ip, average_delay, min_delay, max_delay, delay_var, pkt_loss, residual_bw, average_bw, utilized_bw = struct.unpack_from(cls._FOOTER_FMT, buf, offset)
        offset += cls.FOOTER_SIZE
        remote_ip = addrconv.ipv4.bin_to_text(remote_ip)
        return (cls(lp_status, te_metric, max_bw, max_reserved_bw, unreserved_bw, admin_group, remote_as, remote_ip, average_delay, min_delay, max_delay, delay_var, pkt_loss, residual_bw, average_bw, utilized_bw), buf[offset:])

    def serialize(self):
        buf = struct.pack(self._HEADER_FMT, self.lp_status, self.te_metric, self.max_bw, self.max_reserved_bw, len(self.unreserved_bw))
        for u_bw in self.unreserved_bw:
            buf += struct.pack(self._REPEATED_FMT, u_bw)
        remote_ip = addrconv.ipv4.text_to_bin(self.remote_ip)
        buf += struct.pack(self._FOOTER_FMT, self.admin_group, self.remote_as, remote_ip, self.average_delay, self.min_delay, self.max_delay, self.delay_var, self.pkt_loss, self.residual_bw, self.average_bw, self.utilized_bw)
        return buf