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
class _ClientChannel(comm.ClientChannel):
    """Our protocol, layered on the basic primitives in comm.ClientChannel"""

    def __init__(self, sock, context):
        self.log = logging.getLogger(context.conf.logger_name)
        super(_ClientChannel, self).__init__(sock)
        self.exchange_ping()

    def exchange_ping(self):
        try:
            reply = self.send_recv((comm.Message.PING.value,))
            success = reply[0] == comm.Message.PONG
        except Exception as e:
            self.log.exception('Error while sending initial PING to privsep: %s', e)
            success = False
        if not success:
            msg = _('Privsep daemon failed to start')
            self.log.critical(msg)
            raise FailedToDropPrivileges(msg)

    def remote_call(self, name, args, kwargs, timeout):
        result = self.send_recv((comm.Message.CALL.value, name, args, kwargs), timeout)
        if result[0] == comm.Message.RET:
            return result[1]
        elif result[0] == comm.Message.ERR:
            exc_type = importutils.import_class(result[1])
            raise exc_type(*result[2])
        else:
            raise ProtocolError(_('Unexpected response: %r') % result)

    def out_of_band(self, msg):
        if msg[0] == comm.Message.LOG:
            message = {encodeutils.safe_decode(k): v for k, v in msg[1].items()}
            record = pylogging.makeLogRecord(message)
            if self.log.isEnabledFor(record.levelno):
                self.log.logger.handle(record)
        else:
            self.log.warning('Ignoring unexpected OOB message from privileged process: %r', msg)