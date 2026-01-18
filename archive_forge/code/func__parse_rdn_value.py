import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def _parse_rdn_value(value: memoryview) -> t.Optional[t.Tuple[bytes, int, bool]]:
    if (hex_match := _RDN_VALUE_HEXSTRING_PATTERN.match(value)):
        full_value = hex_match.group(0)
        more_rdns = full_value.endswith(b'+')
        b_value = base64.b16decode(hex_match.group(1).upper())
        return (b_value, len(full_value), more_rdns)
    read = 0
    new_value = bytearray()
    found_spaces = 0
    total_len = len(value)
    while read < total_len:
        current_value = value[read]
        current_char = chr(current_value)
        read += 1
        if current_char == ' ':
            if new_value:
                found_spaces += 1
            continue
        if current_char in [',', '+']:
            break
        if found_spaces:
            new_value += b' ' * found_spaces
            found_spaces = 0
        if current_char == '#' and (not new_value):
            remaining = value[read - 1:].tobytes().decode('utf-8', errors='surrogateescape')
            raise AnsibleFilterError(f"Found leading # for attribute value but does not match hexstring format at '{remaining}'")
        elif current_char in ['\x00', '"', ';', '<', '>']:
            remaining = value[read - 1:].tobytes().decode('utf-8', errors='surrogateescape')
            raise AnsibleFilterError(f"Found unescaped character '{current_char}' in attribute value at '{remaining}'")
        elif current_char == '\\':
            if (escape_match := _RDN_VALUE_ESCAPE_PATTERN.match(value, pos=read)):
                if (literal_value := escape_match.group('literal')):
                    new_value += literal_value
                    read += 1
                else:
                    new_value += base64.b16decode(escape_match.group('hex').upper())
                    read += 2
            else:
                remaining = value[read - 1:].tobytes().decode('utf-8', errors='surrogateescape')
                raise AnsibleFilterError(f"Found invalid escape sequence in attribute value at '{remaining}")
        else:
            new_value.append(current_value)
    if new_value:
        return (bytes(new_value), read, current_char == '+')
    else:
        return None