import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class SubprocessVendor(SSHVendor):
    """Abstract base class for vendors that use pipes to a subprocess."""
    _stderr_target = None

    @staticmethod
    def _check_hostname(arg):
        if arg.startswith('-'):
            raise StrangeHostname(hostname=arg)

    def _connect(self, argv):
        try:
            my_sock, subproc_sock = socket.socketpair()
            osutils.set_fd_cloexec(my_sock)
        except (AttributeError, OSError):
            stdin = stdout = subprocess.PIPE
            my_sock, subproc_sock = (None, None)
        else:
            stdin = stdout = subproc_sock
        proc = subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=self._stderr_target, bufsize=0, **os_specific_subprocess_params())
        if subproc_sock is not None:
            subproc_sock.close()
        return SSHSubprocessConnection(proc, sock=my_sock)

    def connect_sftp(self, username, password, host, port):
        try:
            argv = self._get_vendor_specific_argv(username, host, port, subsystem='sftp')
            sock = self._connect(argv)
            return SFTPClient(SocketAsChannelAdapter(sock))
        except _ssh_connection_errors as e:
            self._raise_connection_error(host, port=port, orig_error=e)

    def connect_ssh(self, username, password, host, port, command):
        try:
            argv = self._get_vendor_specific_argv(username, host, port, command=command)
            return self._connect(argv)
        except _ssh_connection_errors as e:
            self._raise_connection_error(host, port=port, orig_error=e)

    def _get_vendor_specific_argv(self, username, host, port, subsystem=None, command=None):
        """Returns the argument list to run the subprocess with.

        Exactly one of 'subsystem' and 'command' must be specified.
        """
        raise NotImplementedError(self._get_vendor_specific_argv)