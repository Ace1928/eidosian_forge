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
def _parse_section_v2(self, data):
    """ Parses a sequence of packets in data.

        The sequence is terminated by a packet with a field type of EOS
        :param data bytes to be deserialized.
        :return: the rest of data and an array of packet V2
        """
    from pymacaroons.exceptions import MacaroonDeserializationException
    prev_field_type = -1
    packets = []
    while True:
        if len(data) == 0:
            raise MacaroonDeserializationException('section extends past end of buffer')
        rest, packet = self._parse_packet_v2(data)
        if packet.field_type == self._EOS:
            return (rest, packets)
        if packet.field_type <= prev_field_type:
            raise MacaroonDeserializationException('fields out of order')
        packets.append(packet)
        prev_field_type = packet.field_type
        data = rest