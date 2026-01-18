from ctypes import (  # type: ignore[attr-defined]
from socket import AF_INET6, SOCK_STREAM, socket
class SOCKET_ADDRESS_LIST(Structure):
    _fields_ = [('iAddressCount', c_int), ('Address', SOCKET_ADDRESS * ln)]