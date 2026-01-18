from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_bytes
def _pack_octet_integer(value):
    """ Packs an integer value into 1 or multiple octets. """
    octets = bytearray()
    while value:
        octet_value = value & 127
        if len(octets):
            octet_value |= 128
        octets.append(octet_value)
        value >>= 7
    octets.reverse()
    return bytes(octets)