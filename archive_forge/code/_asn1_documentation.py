from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_bytes
Pack the value into an ASN.1 data structure.

    The structure for an ASN.1 element is

    | Identifier Octet(s) | Length Octet(s) | Data Octet(s) |
    