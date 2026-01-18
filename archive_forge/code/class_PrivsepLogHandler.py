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
class PrivsepLogHandler(pylogging.Handler):

    def __init__(self, channel, processName=None):
        super(PrivsepLogHandler, self).__init__()
        self.channel = channel
        self.processName = processName

    def emit(self, record):
        if self.processName:
            record.processName = self.processName
        data = dict(record.__dict__)
        if record.exc_info:
            if not record.exc_text:
                fmt = self.formatter or pylogging.Formatter()
                data['exc_text'] = fmt.formatException(record.exc_info)
            data['exc_info'] = None
        data['msg'] = record.getMessage()
        data['args'] = ()
        self.channel.send((None, (comm.Message.LOG, data)))