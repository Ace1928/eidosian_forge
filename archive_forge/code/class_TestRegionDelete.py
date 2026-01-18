import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegionDelete(TestRegion):

    def setUp(self):
        super(TestRegionDelete, self).setUp()
        self.regions_mock.delete.return_value = None
        self.cmd = region.DeleteRegion(self.app, None)

    def test_region_delete_no_options(self):
        arglist = [identity_fakes.region_id]
        verifylist = [('region', [identity_fakes.region_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.regions_mock.delete.assert_called_with(identity_fakes.region_id)
        self.assertIsNone(result)