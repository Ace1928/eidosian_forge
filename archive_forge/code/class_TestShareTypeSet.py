from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeSet(TestShareType):

    def setUp(self):
        super(TestShareTypeSet, self).setUp()
        self.share_type = manila_fakes.FakeShareType.create_one_sharetype(methods={'set_keys': None, 'update': None})
        self.shares_mock.get.return_value = self.share_type
        self.cmd = osc_share_types.SetShareType(self.app, None)

    def test_share_type_set_extra_specs(self):
        arglist = [self.share_type.id, '--extra-specs', 'snapshot_support=true']
        verifylist = [('share_type', self.share_type.id), ('extra_specs', ['snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_type.set_keys.assert_called_with({'snapshot_support': 'True'})
        self.assertIsNone(result)

    def test_share_type_set_name(self):
        arglist = [self.share_type.id, '--name', 'new name']
        verifylist = [('share_type', self.share_type.id), ('name', 'new name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_type.update.assert_called_with(name='new name')
        self.assertIsNone(result)

    def test_share_type_set_description(self):
        arglist = [self.share_type.id, '--description', 'new description']
        verifylist = [('share_type', self.share_type.id), ('description', 'new description')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_type.update.assert_called_with(description='new description')
        self.assertIsNone(result)

    def test_share_type_set_visibility(self):
        arglist = [self.share_type.id, '--public', 'false']
        verifylist = [('share_type', self.share_type.id), ('public', 'false')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_type.update.assert_called_with(is_public=False)
        self.assertIsNone(result)

    def test_share_type_set_extra_specs_exception(self):
        arglist = [self.share_type.id, '--extra-specs', 'snapshot_support=true']
        verifylist = [('share_type', self.share_type.id), ('extra_specs', ['snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_type.set_keys.side_effect = BadRequest()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_type_set_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.49')
        arglist = [self.share_type.id, '--name', 'new name']
        verifylist = [('share_type', self.share_type.id), ('name', 'new name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)