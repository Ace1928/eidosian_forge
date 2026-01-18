import socket
import threading
from kombu.common import ignore_errors
from kombu.utils.encoding import safe_str
from celery.utils.collections import AttributeDict
from celery.utils.functional import pass1
from celery.utils.log import get_logger
from . import control
def _close_channel(self, c):
    if self.node and self.node.channel:
        ignore_errors(c, self.node.channel.close)