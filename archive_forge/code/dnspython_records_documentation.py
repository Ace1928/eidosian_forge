from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import binary_type

    Convert a DNSPython record data object to a Python dictionary.

    Code borrowed from community.general's dig looup plugin.

    If ``to_unicode=True``, all strings will be converted to Unicode/UTF-8 strings.

    If ``add_synthetic=True``, for some record types additional fields are added.
    For TXT and SPF records, ``value`` contains the concatenated strings, for example.
    