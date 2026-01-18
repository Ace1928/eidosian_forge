import collections
import contextlib
import logging
import socket
import threading
from oslo_config import cfg
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class _Attachment(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.attachments = set()

    def add_attachment(self, attachment_id, host):
        self.attachments.add((attachment_id, host))

    def remove_attachment(self, attachment_id, host):
        self.attachments.remove((attachment_id, host))

    def in_use(self):
        return len(self.attachments) > 0