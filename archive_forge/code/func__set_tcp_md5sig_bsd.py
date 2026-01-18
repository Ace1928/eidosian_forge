import platform
import socket
import struct
from os_ken.lib import sockaddr
def _set_tcp_md5sig_bsd(s, _addr, _key):
    tcp_md5sig = struct.pack('I', 1)
    s.setsockopt(socket.IPPROTO_TCP, TCP_MD5SIG_BSD, tcp_md5sig)