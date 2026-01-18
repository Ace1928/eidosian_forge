from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLockList(TestResourceLock):

    def setUp(self):
        super(TestResourceLockList, self).setUp()
        self.locks = manila_fakes.FakeResourceLock.create_locks(count=2)
        self.locks_mock.list.return_value = self.locks
        self.values = (oscutils.get_dict_properties(m._info, DETAIL_COLUMNS) for m in self.locks)
        self.cmd = osc_resource_locks.ListResourceLock(self.app, None)

    def test_share_lock_list(self):
        arglist = ['--detailed']
        verifylist = [('detailed', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.locks_mock.list.assert_called_with(search_opts={'all_projects': False, 'project_id': None, 'user_id': None, 'id': None, 'resource_id': None, 'resource_type': None, 'resource_action': None, 'lock_context': None, 'created_before': None, 'created_since': None, 'limit': None, 'offset': None}, sort_key=None, sort_dir=None)
        self.assertEqual(sorted(DETAIL_COLUMNS), sorted(columns))
        actual_data = [sorted(d) for d in data]
        expected_data = [sorted(v) for v in self.values]
        self.assertEqual(actual_data, expected_data)