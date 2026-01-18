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
class _ZebraInterface(_ZebraMessageBody, metaclass=abc.ABCMeta):
    """
    Base class for ZEBRA_INTERFACE_ADD, ZEBRA_INTERFACE_DELETE,
    ZEBRA_INTERFACE_UP and ZEBRA_INTERFACE_DOWN message body.
    """
    _HEADER_FMT = '!%dsIBQIIIII' % INTERFACE_NAMSIZE
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _V3_HEADER_FMT = '!%dsIBQIIIIII' % INTERFACE_NAMSIZE
    V3_HEADER_SIZE = struct.calcsize(_V3_HEADER_FMT)
    _V4_HEADER_FMT_2_0 = '!%dsIBQBBIIIIII' % INTERFACE_NAMSIZE
    V4_HEADER_SIZE_2_0 = struct.calcsize(_V4_HEADER_FMT_2_0)
    _V4_HEADER_FMT_3_0 = '!%dsIBQBBIIIIIII' % INTERFACE_NAMSIZE
    V4_HEADER_SIZE_3_0 = struct.calcsize(_V4_HEADER_FMT_3_0)
    _LP_STATE_FMT = '!?'
    LP_STATE_SIZE = struct.calcsize(_LP_STATE_FMT)

    def __init__(self, ifname=None, ifindex=None, status=None, if_flags=None, ptm_enable=None, ptm_status=None, metric=None, speed=None, ifmtu=None, ifmtu6=None, bandwidth=None, ll_type=None, hw_addr_len=0, hw_addr=None, link_params=None):
        super(_ZebraInterface, self).__init__()
        self.ifname = ifname
        self.ifindex = ifindex
        self.status = status
        self.if_flags = if_flags
        self.ptm_enable = ptm_enable
        self.ptm_status = ptm_status
        self.metric = metric
        self.speed = speed
        self.ifmtu = ifmtu
        self.ifmtu6 = ifmtu6
        self.bandwidth = bandwidth
        self.ll_type = ll_type
        self.hw_addr_lenght = hw_addr_len
        hw_addr = hw_addr or b''
        self.hw_addr = hw_addr
        assert isinstance(link_params, InterfaceLinkParams) or link_params is None
        self.link_params = link_params

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        ptm_enable = None
        ptm_status = None
        speed = None
        ll_type = None
        if version <= 2:
            ifname, ifindex, status, if_flags, metric, ifmtu, ifmtu6, bandwidth, hw_addr_len = struct.unpack_from(cls._HEADER_FMT, buf)
            rest = buf[cls.HEADER_SIZE:]
        elif version == 3:
            ifname, ifindex, status, if_flags, metric, ifmtu, ifmtu6, bandwidth, ll_type, hw_addr_len = struct.unpack_from(cls._V3_HEADER_FMT, buf)
            rest = buf[cls.V3_HEADER_SIZE:]
        elif version == 4:
            if _is_frr_version_ge(_FRR_VERSION_3_0):
                ifname, ifindex, status, if_flags, ptm_enable, ptm_status, metric, speed, ifmtu, ifmtu6, bandwidth, ll_type, hw_addr_len = struct.unpack_from(cls._V4_HEADER_FMT_3_0, buf)
                rest = buf[cls.V4_HEADER_SIZE_3_0:]
            elif _is_frr_version_ge(_FRR_VERSION_2_0):
                ifname, ifindex, status, if_flags, ptm_enable, ptm_status, metric, ifmtu, ifmtu6, bandwidth, ll_type, hw_addr_len = struct.unpack_from(cls._V4_HEADER_FMT_2_0, buf)
                rest = buf[cls.V4_HEADER_SIZE_2_0:]
            else:
                raise struct.error('Unsupported FRRouting version: %s' % CONF['zapi'].frr_version)
        else:
            raise struct.error('Unsupported Zebra protocol version: %d' % version)
        ifname = str(str(ifname.strip(b'\x00'), 'ascii'))
        hw_addr_len = min(hw_addr_len, INTERFACE_HWADDR_MAX)
        hw_addr_bin = rest[:hw_addr_len]
        rest = rest[hw_addr_len:]
        if 0 < hw_addr_len < 7:
            hw_addr = addrconv.mac.bin_to_text(hw_addr_bin + b'\x00' * (6 - hw_addr_len))
        else:
            hw_addr = hw_addr_bin
        if not rest:
            return cls(ifname, ifindex, status, if_flags, ptm_enable, ptm_status, metric, speed, ifmtu, ifmtu6, bandwidth, ll_type, hw_addr_len, hw_addr)
        link_param_state, = struct.unpack_from(cls._LP_STATE_FMT, rest)
        rest = rest[cls.LP_STATE_SIZE:]
        if link_param_state:
            link_params, rest = InterfaceLinkParams.parse(rest)
        else:
            link_params = None
        return cls(ifname, ifindex, status, if_flags, ptm_enable, ptm_status, metric, speed, ifmtu, ifmtu6, bandwidth, ll_type, hw_addr_len, hw_addr, link_params)

    def serialize(self, version=_DEFAULT_VERSION):
        if self.ifname is None:
            return b''
        if netaddr.valid_mac(self.hw_addr):
            hw_addr_len = 6
            hw_addr = addrconv.mac.text_to_bin(self.hw_addr)
        else:
            hw_addr_len = len(self.hw_addr)
            hw_addr = self.hw_addr
        if version <= 2:
            return struct.pack(self._HEADER_FMT, self.ifname.encode('ascii'), self.ifindex, self.status, self.if_flags, self.metric, self.ifmtu, self.ifmtu6, self.bandwidth, hw_addr_len) + hw_addr
        elif version == 3:
            buf = struct.pack(self._V3_HEADER_FMT, self.ifname.encode('ascii'), self.ifindex, self.status, self.if_flags, self.metric, self.ifmtu, self.ifmtu6, self.bandwidth, self.ll_type, hw_addr_len) + hw_addr
        elif version == 4:
            if _is_frr_version_ge(_FRR_VERSION_3_0):
                buf = struct.pack(self._V4_HEADER_FMT_3_0, self.ifname.encode('ascii'), self.ifindex, self.status, self.if_flags, self.ptm_enable, self.ptm_status, self.metric, self.speed, self.ifmtu, self.ifmtu6, self.bandwidth, self.ll_type, hw_addr_len) + hw_addr
            elif _is_frr_version_ge(_FRR_VERSION_2_0):
                buf = struct.pack(self._V4_HEADER_FMT_2_0, self.ifname.encode('ascii'), self.ifindex, self.status, self.if_flags, self.ptm_enable, self.ptm_status, self.metric, self.ifmtu, self.ifmtu6, self.bandwidth, self.ll_type, hw_addr_len) + hw_addr
            else:
                raise ValueError('Unsupported FRRouting version: %s' % CONF['zapi'].frr_version)
        else:
            raise ValueError('Unsupported Zebra protocol version: %d' % version)
        if isinstance(self.link_params, InterfaceLinkParams):
            buf += struct.pack(self._LP_STATE_FMT, True)
            buf += self.link_params.serialize()
        else:
            buf += struct.pack(self._LP_STATE_FMT, False)
        return buf