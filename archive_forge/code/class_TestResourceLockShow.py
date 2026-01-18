from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLockShow(TestResourceLock):

    def setUp(self):
        super(TestResourceLockShow, self).setUp()
        self.lock = manila_fakes.FakeResourceLock.create_one_lock()
        self.locks_mock.get.return_value = self.lock
        self.cmd = osc_resource_locks.ShowResourceLock(self.app, None)
        self.data = self.lock._info.values()
        self.columns = list(self.lock._info.keys())

    def test_share_lock_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_lock_show(self):
        arglist = [self.lock.id]
        verifylist = [('lock', self.lock.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.locks_mock.get.assert_called_with(self.lock.id)
        self.assertEqual(len(self.columns), len(columns))
        self.assertCountEqual(sorted(self.data), sorted(data))