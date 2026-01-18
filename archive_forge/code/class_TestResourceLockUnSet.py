from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLockUnSet(TestResourceLock):

    def setUp(self):
        super(TestResourceLockUnSet, self).setUp()
        self.lock = manila_fakes.FakeResourceLock.create_one_lock()
        self.lock.update = mock.Mock()
        self.locks_mock.get.return_value = self.lock
        self.cmd = osc_resource_locks.UnsetResourceLock(self.app, None)

    def test_share_lock_unset_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_lock_unset(self):
        arglist = [self.lock.id, '--lock-reason']
        verifylist = [('lock', self.lock.id), ('lock_reason', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.locks_mock.update.assert_called_with(self.lock.id, lock_reason=None)