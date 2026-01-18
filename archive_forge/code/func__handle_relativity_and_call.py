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
def _handle_relativity_and_call(function: Callable[[Name, Name, bool], Name], name: Name, origin: Name, prefix_ok: bool) -> Name:
    if not origin.is_absolute():
        raise NeedAbsoluteNameOrOrigin
    relative = not name.is_absolute()
    if relative:
        name = name.derelativize(origin)
    elif not name.is_subdomain(origin):
        raise NeedSubdomainOfOrigin
    result_name = function(name, origin, prefix_ok)
    if relative:
        result_name = result_name.relativize(origin)
    return result_name