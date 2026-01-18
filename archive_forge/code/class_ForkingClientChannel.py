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
class ForkingClientChannel(_ClientChannel):

    def __init__(self, context):
        """Start privsep daemon using fork()

        Assumes we already have required privileges.
        """
        sock_a, sock_b = socket.socketpair()
        for s in (sock_a, sock_b):
            s.setblocking(True)
            set_cloexec(s)
        for f in (sys.stdout, sys.stderr):
            f.flush()
        if os.fork() == 0:
            un_monkey_patch()
            channel = comm.ServerChannel(sock_b)
            sock_a.close()
            replace_logging(PrivsepLogHandler(channel, processName=str(context)))
            Daemon(channel, context=context).run()
            LOG.debug('privsep daemon exiting')
            os._exit(0)
        sock_b.close()
        super(ForkingClientChannel, self).__init__(sock_a, context)