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
def _append_packet(self, data, field_type, packet_data=None):
    _encode_uvarint(data, field_type)
    if field_type != self._EOS:
        _encode_uvarint(data, len(packet_data))
        data.extend(packet_data)