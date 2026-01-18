import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def _parse_rdn_type(value: memoryview) -> t.Optional[t.Tuple[bytes, int]]:
    if (match := _RDN_TYPE_PATTERN.match(value)):
        return (match.group(1), len(match.group(0)))
    return None