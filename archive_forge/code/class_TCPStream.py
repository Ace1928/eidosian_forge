import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
class TCPStream(Stream):

    @staticmethod
    def needs_probes():
        return True

    @staticmethod
    def _open(suffix, dscp):
        error, sock = ovs.socket_util.inet_open_active(socket.SOCK_STREAM, suffix, 0, dscp)
        if not error:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except socket.error as e:
                sock.close()
                return (ovs.socket_util.get_exception_errno(e), None)
        return (error, sock)