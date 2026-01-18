import socket
import sys
from ctypes import (
from ctypes.util import find_library
from socket import AF_INET, AF_INET6, inet_ntop
from typing import Any, List, Tuple
from twisted.python.compat import nativeString
class ifaddrs(Structure):
    pass