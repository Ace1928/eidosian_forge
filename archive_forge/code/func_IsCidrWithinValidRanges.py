from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def IsCidrWithinValidRanges(cidr):
    """Checks if a given CIDR block is contained within a list of valid CIDR ranges."""
    rfc_1918_spaces = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
    rfc_6598_spaces = ['100.64.0.0/10']
    rfc_6890_spaces = ['192.0.0.0/24']
    rfc_5737_spaces = ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24']
    rfc_7526_spaces = ['192.88.99.0/24']
    rfc_2544_spaces = ['198.18.0.0/15']
    valid_cidr_ranges = rfc_1918_spaces + rfc_6598_spaces + rfc_6890_spaces + rfc_5737_spaces + rfc_7526_spaces + rfc_2544_spaces
    cidr_block = ipaddress.IPv4Network(cidr)
    for valid_range in valid_cidr_ranges:
        if cidr_block.subnet_of(ipaddress.IPv4Network(valid_range)):
            return True
    return False