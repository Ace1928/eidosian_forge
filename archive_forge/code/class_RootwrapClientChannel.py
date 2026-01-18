from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
class RootwrapClientChannel(_ClientChannel):

    def __init__(self, context):
        """Start privsep daemon using exec()

        Uses sudo/rootwrap to gain privileges.
        """
        listen_sock = socket.socket(socket.AF_UNIX)
        tmpdir = tempfile.mkdtemp()
        try:
            sockpath = os.path.join(tmpdir, 'privsep.sock')
            listen_sock.bind(sockpath)
            listen_sock.listen(1)
            cmd = context.helper_command(sockpath)
            LOG.info('Running privsep helper: %s', cmd)
            proc = subprocess.Popen(cmd, shell=False, stderr=_fd_logger())
            if proc.wait() != 0:
                msg = 'privsep helper command exited non-zero (%s)' % proc.returncode
                LOG.critical(msg)
                raise FailedToDropPrivileges(msg)
            LOG.info('Spawned new privsep daemon via rootwrap')
            sock, _addr = listen_sock.accept()
            LOG.debug('Accepted privsep connection to %s', sockpath)
        finally:
            listen_sock.close()
            try:
                os.unlink(sockpath)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            os.rmdir(tmpdir)
        super(RootwrapClientChannel, self).__init__(sock, context)