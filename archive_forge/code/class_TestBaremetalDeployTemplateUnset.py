import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalDeployTemplateUnset(TestBaremetalDeployTemplate):

    def setUp(self):
        super(TestBaremetalDeployTemplateUnset, self).setUp()
        self.baremetal_mock.deploy_template.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.DEPLOY_TEMPLATE), loaded=True)
        self.cmd = baremetal_deploy_template.UnsetBaremetalDeployTemplate(self.app, None)

    def test_baremetal_deploy_template_unset_extra(self):
        arglist = [baremetal_fakes.baremetal_deploy_template_uuid, '--extra', 'key1']
        verifylist = [('template', baremetal_fakes.baremetal_deploy_template_uuid), ('extra', ['key1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.deploy_template.update.assert_called_once_with(baremetal_fakes.baremetal_deploy_template_uuid, [{'path': '/extra/key1', 'op': 'remove'}])

    def test_baremetal_deploy_template_unset_multiple_extras(self):
        arglist = [baremetal_fakes.baremetal_deploy_template_uuid, '--extra', 'key1', '--extra', 'key2']
        verifylist = [('template', baremetal_fakes.baremetal_deploy_template_uuid), ('extra', ['key1', 'key2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.deploy_template.update.assert_called_once_with(baremetal_fakes.baremetal_deploy_template_uuid, [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])

    def test_baremetal_deploy_template_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_deploy_template_unset_no_property(self):
        uuid = baremetal_fakes.baremetal_deploy_template_uuid
        arglist = [uuid]
        verifylist = [('template', uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.deploy_template.update.called)