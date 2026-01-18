import socket
import sys
from ctypes import (
from ctypes.util import find_library
from socket import AF_INET, AF_INET6, inet_ntop
from typing import Any, List, Tuple
from twisted.python.compat import nativeString

    Return a list of strings in colon-hex format representing all the link local
    IPv6 addresses available on the system, as reported by I{getifaddrs(3)}.
    