import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalDeployTemplate(TestBaremetalDeployTemplate):

    def setUp(self):
        super(TestCreateBaremetalDeployTemplate, self).setUp()
        self.baremetal_mock.deploy_template.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.DEPLOY_TEMPLATE), loaded=True)
        self.cmd = baremetal_deploy_template.CreateBaremetalDeployTemplate(self.app, None)

    def test_baremetal_deploy_template_create(self):
        arglist = [baremetal_fakes.baremetal_deploy_template_name, '--steps', baremetal_fakes.baremetal_deploy_template_steps]
        verifylist = [('name', baremetal_fakes.baremetal_deploy_template_name), ('steps', baremetal_fakes.baremetal_deploy_template_steps)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'name': baremetal_fakes.baremetal_deploy_template_name, 'steps': json.loads(baremetal_fakes.baremetal_deploy_template_steps)}
        self.baremetal_mock.deploy_template.create.assert_called_once_with(**args)

    def test_baremetal_deploy_template_create_uuid(self):
        arglist = [baremetal_fakes.baremetal_deploy_template_name, '--steps', baremetal_fakes.baremetal_deploy_template_steps, '--uuid', baremetal_fakes.baremetal_deploy_template_uuid]
        verifylist = [('name', baremetal_fakes.baremetal_deploy_template_name), ('steps', baremetal_fakes.baremetal_deploy_template_steps), ('uuid', baremetal_fakes.baremetal_deploy_template_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'name': baremetal_fakes.baremetal_deploy_template_name, 'steps': json.loads(baremetal_fakes.baremetal_deploy_template_steps), 'uuid': baremetal_fakes.baremetal_deploy_template_uuid}
        self.baremetal_mock.deploy_template.create.assert_called_once_with(**args)

    def test_baremetal_deploy_template_create_no_name(self):
        arglist = ['--steps', baremetal_fakes.baremetal_deploy_template_steps]
        verifylist = [('steps', baremetal_fakes.baremetal_deploy_template_steps)]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertFalse(self.baremetal_mock.deploy_template.create.called)

    def test_baremetal_deploy_template_create_no_steps(self):
        arglist = ['--name', baremetal_fakes.baremetal_deploy_template_name]
        verifylist = [('name', baremetal_fakes.baremetal_deploy_template_name)]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertFalse(self.baremetal_mock.deploy_template.create.called)