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
class TestShareGroupTypeUnset(TestShareGroupType):

    def setUp(self):
        super(TestShareGroupTypeUnset, self).setUp()
        self.share_group_type = manila_fakes.FakeShareGroupType.create_one_share_group_type(methods={'unset_keys': None})
        self.sgt_mock.get.return_value = self.share_group_type
        self.cmd = osc_share_group_types.UnsetShareGroupType(self.app, None)

    def test_share_group_type_unset_extra_specs(self):
        arglist = [self.share_group_type.id, 'consistent_snapshot_support']
        verifylist = [('share_group_type', self.share_group_type.id), ('group_specs', ['consistent_snapshot_support'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_group_type.unset_keys.assert_called_with(['consistent_snapshot_support'])
        self.assertIsNone(result)

    def test_share_group_type_unset_exception(self):
        arglist = [self.share_group_type.id, 'snapshot_support']
        verifylist = [('share_group_type', self.share_group_type.id), ('group_specs', ['snapshot_support'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_group_type.unset_keys.side_effect = NotFound()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)