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
def _close_stdio(self):
    with open(os.devnull, 'w+') as devnull:
        os.dup2(devnull.fileno(), StdioFd.STDIN)
        os.dup2(devnull.fileno(), StdioFd.STDOUT)