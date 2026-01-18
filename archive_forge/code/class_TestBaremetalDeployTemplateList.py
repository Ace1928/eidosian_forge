import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalDeployTemplateList(TestBaremetalDeployTemplate):

    def setUp(self):
        super(TestBaremetalDeployTemplateList, self).setUp()
        self.baremetal_mock.deploy_template.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.DEPLOY_TEMPLATE), loaded=True)]
        self.cmd = baremetal_deploy_template.ListBaremetalDeployTemplate(self.app, None)

    def test_baremetal_deploy_template_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.deploy_template.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Name')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_deploy_template_uuid, baremetal_fakes.baremetal_deploy_template_name),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_deploy_template_list_long(self):
        arglist = ['--long']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.deploy_template.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Name', 'Steps', 'Extra', 'Created At', 'Updated At')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_deploy_template_uuid, baremetal_fakes.baremetal_deploy_template_name, baremetal_fakes.baremetal_deploy_template_steps, baremetal_fakes.baremetal_deploy_template_extra, '', ''),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_deploy_template_list_fields(self):
        arglist = ['--fields', 'uuid', 'steps']
        verifylist = [('fields', [['uuid', 'steps']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'steps')}
        self.baremetal_mock.deploy_template.list.assert_called_with(**kwargs)

    def test_baremetal_deploy_template_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'name', '--fields', 'steps']
        verifylist = [('fields', [['uuid', 'name'], ['steps']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'name', 'steps')}
        self.baremetal_mock.deploy_template.list.assert_called_with(**kwargs)

    def test_baremetal_deploy_template_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)