import copy
import encodings.idna  # type: ignore
import functools
import struct
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import dns._features
import dns.enum
import dns.exception
import dns.immutable
import dns.wire
def _absolute_predecessor(name: Name, origin: Name, prefix_ok: bool) -> Name:
    if name == origin:
        return _pad_to_max_name(name)
    least_significant_label = name[0]
    if least_significant_label == _MINIMAL_OCTET:
        return name.parent()
    least_octet = least_significant_label[-1]
    suffix_labels = name.labels[1:]
    if least_octet == _MINIMAL_OCTET_VALUE:
        new_labels = [least_significant_label[:-1]]
    else:
        octets = bytearray(least_significant_label)
        octet = octets[-1]
        if octet == _LEFT_SQUARE_BRACKET_VALUE:
            octet = _AT_SIGN_VALUE
        else:
            octet -= 1
        octets[-1] = octet
        least_significant_label = bytes(octets)
        new_labels = [_pad_to_max_label(least_significant_label, suffix_labels)]
    new_labels.extend(suffix_labels)
    name = Name(new_labels)
    if prefix_ok:
        return _pad_to_max_name(name)
    else:
        return name