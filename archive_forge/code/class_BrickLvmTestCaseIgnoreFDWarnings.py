from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
class BrickLvmTestCaseIgnoreFDWarnings(BrickLvmTestCase):

    def setUp(self):
        self.configuration = mock.Mock()
        self.configuration.lvm_suppress_fd_warnings = True
        super(BrickLvmTestCaseIgnoreFDWarnings, self).setUp()