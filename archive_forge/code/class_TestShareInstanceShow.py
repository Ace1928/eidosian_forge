from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceShow(TestShareInstance):

    def setUp(self):
        super(TestShareInstanceShow, self).setUp()
        self.share_instance = manila_fakes.FakeShareInstance.create_one_share_instance()
        self.instances_mock.get.return_value = self.share_instance
        self.export_locations = [manila_fakes.FakeShareExportLocation.create_one_export_location() for i in range(2)]
        self.share_instance_export_locations_mock.list.return_value = self.export_locations
        self.cmd = osc_share_instances.ShareInstanceShow(self.app, None)
        self.data = tuple(self.share_instance._info.values())
        self.columns = tuple(self.share_instance._info.keys())

    def test_share_instance_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_instance_show(self):
        expected_columns = tuple(self.share_instance._info.keys())
        expected_data_dic = tuple()
        for column in expected_columns:
            expected_data_dic += (self.share_instance._info[column],)
        expected_columns += ('export_locations',)
        expected_data_dic += (dict(self.export_locations[0]),)
        cliutils.convert_dict_list_to_string = mock.Mock()
        cliutils.convert_dict_list_to_string.return_value = dict(self.export_locations[0])
        arglist = [self.share_instance.id]
        verifylist = [('instance', self.share_instance.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.instances_mock.get.assert_called_with(self.share_instance.id)
        self.assertCountEqual(expected_columns, columns)
        self.assertCountEqual(expected_data_dic, data)