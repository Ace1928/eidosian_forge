from troveclient.osc.v1 import database_flavors
from troveclient.tests.osc.v1 import fakes
class TestFlavorShow(TestFlavors):
    values = (1, 'm1.tiny', 512)

    def setUp(self):
        super(TestFlavorShow, self).setUp()
        self.cmd = database_flavors.ShowDatabaseFlavor(self.app, None)
        self.data = self.fake_flavors.get_flavors_1()
        self.flavor_client.get.return_value = self.data
        self.columns = ('id', 'name', 'ram')

    def test_flavor_show_defaults(self):
        args = ['m1.tiny']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)