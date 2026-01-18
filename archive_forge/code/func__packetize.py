from __future__ import unicode_literals
import binascii
from collections import namedtuple
import six
import struct
import sys
from base64 import urlsafe_b64encode
from pymacaroons.utils import (
from pymacaroons.serializers.base_serializer import BaseSerializer
from pymacaroons.exceptions import MacaroonSerializationException
def _packetize(self, key, data):
    packet_size = self.PACKET_PREFIX_LENGTH + 2 + len(key) + len(data)
    packet_size_hex = hex(packet_size)[2:]
    if packet_size > 65535:
        raise MacaroonSerializationException('Packet too long for serialization. Max length is 0xFFFF (65535). Packet length: 0x{hex_length} ({length}) Key: {key}'.format(key=key, hex_length=packet_size_hex, length=packet_size))
    header = packet_size_hex.zfill(4).encode('ascii')
    packet_content = key + b' ' + convert_to_bytes(data) + b'\n'
    packet = struct.pack(convert_to_bytes('4s%ds' % len(packet_content)), header, packet_content)
    return packet