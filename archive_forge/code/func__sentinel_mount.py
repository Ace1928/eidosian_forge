import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
def _sentinel_mount(self):
    self.m.mount(mock.sentinel.fstype, mock.sentinel.export, mock.sentinel.vol, mock.sentinel.mountpoint, mock.sentinel.host, mock.sentinel.rootwrap_helper, [mock.sentinel.option1, mock.sentinel.option2])