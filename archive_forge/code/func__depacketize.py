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
def _depacketize(self, packet):
    key = packet.split(b' ')[0]
    value = packet[len(key) + 1:-1]
    return (key, value)