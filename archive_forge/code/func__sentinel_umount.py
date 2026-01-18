import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
def _sentinel_umount(self):
    self.m.umount(mock.sentinel.vol, mock.sentinel.mountpoint, mock.sentinel.host, mock.sentinel.rootwrap_helper)