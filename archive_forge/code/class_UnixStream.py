import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
class UnixStream(Stream):

    @staticmethod
    def needs_probes():
        return False

    @staticmethod
    def _open(suffix, dscp):
        connect_path = suffix
        return ovs.socket_util.make_unix_socket(socket.SOCK_STREAM, True, None, connect_path)