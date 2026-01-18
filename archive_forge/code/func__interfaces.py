import socket
import sys
from ctypes import (
from ctypes.util import find_library
from socket import AF_INET, AF_INET6, inet_ntop
from typing import Any, List, Tuple
from twisted.python.compat import nativeString
def _interfaces():
    """
    Call C{getifaddrs(3)} and return a list of tuples of interface name, address
    family, and human-readable address representing its results.
    """
    ifaddrs = ifaddrs_p()
    if getifaddrs(pointer(ifaddrs)) < 0:
        raise OSError()
    results = []
    try:
        while ifaddrs:
            if ifaddrs[0].ifa_addr:
                family = ifaddrs[0].ifa_addr[0].sin_family
                if family == AF_INET:
                    addr = cast(ifaddrs[0].ifa_addr, POINTER(sockaddr_in))
                elif family == AF_INET6:
                    addr = cast(ifaddrs[0].ifa_addr, POINTER(sockaddr_in6))
                else:
                    addr = None
                if addr:
                    packed = bytes(addr[0].sin_addr.in_addr[:])
                    packed = _maybeCleanupScopeIndex(family, packed)
                    results.append((ifaddrs[0].ifa_name, family, inet_ntop(family, packed)))
            ifaddrs = ifaddrs[0].ifa_next
    finally:
        freeifaddrs(ifaddrs)
    return results