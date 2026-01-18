import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosHostAddressType(enum.IntEnum):
    ipv4 = 2
    directional = 3
    chaos_net = 5
    xns = 6
    iso = 7
    decnet_phase_iv = 12
    apple_talk_ddp = 16
    netbios = 20
    ipv6 = 26

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosHostAddressType', str]:
        return {KerberosHostAddressType.ipv4: 'IPv4', KerberosHostAddressType.directional: 'Directional', KerberosHostAddressType.chaos_net: 'ChaosNet', KerberosHostAddressType.xns: 'XNS', KerberosHostAddressType.iso: 'ISO', KerberosHostAddressType.decnet_phase_iv: 'DECNET Phase IV', KerberosHostAddressType.apple_talk_ddp: 'AppleTalk DDP', KerberosHostAddressType.netbios: 'NetBios', KerberosHostAddressType.ipv6: 'IPv6'}