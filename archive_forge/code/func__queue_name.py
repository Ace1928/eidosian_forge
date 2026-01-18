import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
def _queue_name(self, topic):
    return '%s_%s' % (self.exchange, topic)