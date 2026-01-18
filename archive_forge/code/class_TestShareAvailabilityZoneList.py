from osc_lib import utils as oscutils
from manilaclient.osc.v2 import availability_zones as osc_availability_zones
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareAvailabilityZoneList(TestAvailabilityZones):
    availability_zones = manila_fakes.FakeShareAvailabilityZones.create_share_availability_zones()
    COLUMNS = ('Id', 'Name', 'Created At', 'Updated At')

    def setUp(self):
        super(TestShareAvailabilityZoneList, self).setUp()
        self.zones_mock.list.return_value = self.availability_zones
        self.cmd = osc_availability_zones.ShareAvailabilityZoneList(self.app, None)
        self.values = (oscutils.get_dict_properties(s._info, self.COLUMNS) for s in self.availability_zones)

    def test_share_list_availability_zone(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.COLUMNS, columns)
        self.assertCountEqual(list(self.values), list(data))