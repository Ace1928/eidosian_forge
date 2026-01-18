import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
@staticmethod
def _expected_sentinel_mount_calls(mountpoint=mock.sentinel.mountpoint):
    return [mock.call('mount', '-t', mock.sentinel.fstype, mock.sentinel.option1, mock.sentinel.option2, mock.sentinel.export, mountpoint, root_helper=mock.sentinel.rootwrap_helper, run_as_root=True)]