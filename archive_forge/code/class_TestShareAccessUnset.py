from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareAccessUnset(TestShareAccess):

    def setUp(self):
        super(TestShareAccessUnset, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.access_rule = manila_fakes.FakeShareAccessRule.create_one_access_rule(attrs={'share_id': self.share.id})
        self.access_rules_mock.get.return_value = self.access_rule
        self.cmd = osc_share_access_rules.UnsetShareAccess(self.app, None)

    def test_access_rule_unset(self):
        arglist = [self.access_rule.id, '--property', 'key1']
        verifylist = [('access_id', self.access_rule.id), ('property', ['key1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.access_rules_mock.unset_metadata.assert_called_with(self.access_rule, ['key1'])
        self.assertIsNone(result)