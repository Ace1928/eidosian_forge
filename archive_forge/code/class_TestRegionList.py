import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegionList(TestRegion):
    columns = ('Region', 'Parent Region', 'Description')
    datalist = ((identity_fakes.region_id, identity_fakes.region_parent_region_id, identity_fakes.region_description),)

    def setUp(self):
        super(TestRegionList, self).setUp()
        self.regions_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGION), loaded=True)]
        self.cmd = region.ListRegion(self.app, None)

    def test_region_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.regions_mock.list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_region_list_parent_region_id(self):
        arglist = ['--parent-region', identity_fakes.region_parent_region_id]
        verifylist = [('parent_region', identity_fakes.region_parent_region_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.regions_mock.list.assert_called_with(parent_region_id=identity_fakes.region_parent_region_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))