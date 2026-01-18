from ctypes import (  # type: ignore[attr-defined]
from socket import AF_INET6, SOCK_STREAM, socket
class SOCKET_ADDRESS(Structure):
    _fields_ = [('lpSockaddr', c_void_p), ('iSockaddrLength', c_int)]