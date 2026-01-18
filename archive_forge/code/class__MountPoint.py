import collections
import contextlib
import logging
import os
import socket
import threading
from oslo_concurrency import processutils
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class _MountPoint(object):
    """A single mountpoint, and the set of attachments in use on it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.attachments = set()

    def add_attachment(self, vol_name, host):
        self.attachments.add((vol_name, host))

    def remove_attachment(self, vol_name, host):
        self.attachments.remove((vol_name, host))

    def in_use(self):
        return len(self.attachments) > 0