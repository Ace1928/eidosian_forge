from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
class InstanceStatusTest(testtools.TestCase):

    def test_constants(self):
        self.assertEqual('ACTIVE', instances.InstanceStatus.ACTIVE)
        self.assertEqual('BLOCKED', instances.InstanceStatus.BLOCKED)
        self.assertEqual('BUILD', instances.InstanceStatus.BUILD)
        self.assertEqual('FAILED', instances.InstanceStatus.FAILED)
        self.assertEqual('REBOOT', instances.InstanceStatus.REBOOT)
        self.assertEqual('RESIZE', instances.InstanceStatus.RESIZE)
        self.assertEqual('SHUTDOWN', instances.InstanceStatus.SHUTDOWN)
        self.assertEqual('RESTART_REQUIRED', instances.InstanceStatus.RESTART_REQUIRED)