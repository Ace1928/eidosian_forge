from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLockDelete(TestResourceLock):

    def setUp(self):
        super(TestResourceLockDelete, self).setUp()
        self.lock = manila_fakes.FakeResourceLock.create_one_lock()
        self.locks_mock.get.return_value = self.lock
        self.lock.delete = mock.Mock()
        self.cmd = osc_resource_locks.DeleteResourceLock(self.app, None)

    def test_share_lock_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_lock_delete(self):
        arglist = [self.lock.id]
        verifylist = [('lock', [self.lock.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.lock.delete.assert_called_once_with()
        self.assertIsNone(result)

    def test_share_lock_delete_multiple(self):
        locks = manila_fakes.FakeResourceLock.create_locks(count=2)
        arglist = [locks[0].id, locks[1].id]
        verifylist = [('lock', [locks[0].id, locks[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.lock.delete.call_count, len(locks))
        self.assertIsNone(result)

    def test_share_lock_delete_exception(self):
        arglist = [self.lock.id]
        verifylist = [('lock', [self.lock.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.lock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)