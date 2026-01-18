from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupTypeShow(TestShareGroupType):

    def setUp(self):
        super(TestShareGroupTypeShow, self).setUp()
        self.share_types = manila_fakes.FakeShareType.create_share_types(count=2)
        formatted_share_types = []
        for st in self.share_types:
            formatted_share_types.append(st.name)
        self.share_group_type = manila_fakes.FakeShareGroupType.create_one_share_group_type(attrs={'share_types': formatted_share_types})
        self.share_group_type_formatted = manila_fakes.FakeShareGroupType.create_one_share_group_type(attrs={'id': self.share_group_type['id'], 'name': self.share_group_type['name'], 'share_types': formatted_share_types})
        formatted_sgt = utils.format_share_group_type(self.share_group_type_formatted)
        self.sgt_mock.get.return_value = self.share_group_type
        self.cmd = osc_share_group_types.ShowShareGroupType(self.app, None)
        self.data = tuple(formatted_sgt.values())
        self.columns = tuple(formatted_sgt.keys())

    def test_share_group_type_show(self):
        arglist = [self.share_group_type.name]
        verifylist = [('share_group_type', self.share_group_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sgt_mock.get.assert_called_with(self.share_group_type)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)