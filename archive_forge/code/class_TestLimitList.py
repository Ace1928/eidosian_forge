from troveclient import common
from troveclient.osc.v1 import database_limits
from troveclient.tests.osc.v1 import fakes
class TestLimitList(TestLimits):
    columns = database_limits.ListDatabaseLimits.columns
    non_absolute_values = (200, 'DELETE', 200, 'MINUTE')

    def setUp(self):
        super(TestLimitList, self).setUp()
        self.cmd = database_limits.ListDatabaseLimits(self.app, None)
        data = [self.fake_limits.get_absolute_limits(), self.fake_limits.get_non_absolute_limits()]
        self.limit_client.list.return_value = common.Paginated(data)

    def test_limit_list_defaults(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.limit_client.list.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.non_absolute_values], data)