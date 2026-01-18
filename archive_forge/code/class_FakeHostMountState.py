import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
class FakeHostMountState:

    def __init__(self):
        self.mountpoints = {mock.sentinel.mountpoint}