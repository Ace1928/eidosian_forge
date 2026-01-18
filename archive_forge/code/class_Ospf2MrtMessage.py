import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
class Ospf2MrtMessage(MrtMessage):
    """
    MRT Message for the OSPFv2 Type.
    """
    _HEADER_FMT = '!4s4s'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _TYPE = {'ascii': ['remote_ip', 'local_ip']}

    def __init__(self, remote_ip, local_ip, ospf_message):
        self.remote_ip = remote_ip
        self.local_ip = local_ip
        assert isinstance(ospf_message, ospf.OSPFMessage)
        self.ospf_message = ospf_message

    @classmethod
    def parse(cls, buf):
        remote_ip, local_ip = struct.unpack_from(cls._HEADER_FMT, buf)
        remote_ip = addrconv.ipv4.bin_to_text(remote_ip)
        local_ip = addrconv.ipv4.bin_to_text(local_ip)
        ospf_message, _, _ = ospf.OSPFMessage.parser(buf[cls.HEADER_SIZE:])
        return cls(remote_ip, local_ip, ospf_message)

    def serialize(self):
        return addrconv.ipv4.text_to_bin(self.remote_ip) + addrconv.ipv4.text_to_bin(self.local_ip) + self.ospf_message.serialize()