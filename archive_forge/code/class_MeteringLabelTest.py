from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.neutron import metering
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class MeteringLabelTest(common.HeatTestCase):

    def setUp(self):
        super(MeteringLabelTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)

    def create_metering_label(self):
        self.mockclient.create_metering_label.return_value = {'metering_label': {'id': '1234'}}
        snippet = template_format.parse(metering_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return metering.MeteringLabel('label', resource_defns['label'], self.stack)

    def test_create(self):
        rsrc = self.create_metering_label()
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})

    def test_create_failed(self):
        self.mockclient.create_metering_label.side_effect = exceptions.NeutronClientException()
        snippet = template_format.parse(metering_template)
        stack = utils.parse_stack(snippet)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = metering.MeteringLabel('label', resource_defns['label'], stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.label: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})

    def test_delete(self):
        rsrc = self.create_metering_label()
        self.mockclient.delete_metering_label.return_value = None
        self.mockclient.show_metering_label.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})
        self.mockclient.delete_metering_label.assert_called_once_with('1234')
        self.mockclient.show_metering_label.assert_called_once_with('1234')

    def test_delete_already_gone(self):
        rsrc = self.create_metering_label()
        self.mockclient.delete_metering_label.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})
        self.mockclient.delete_metering_label.assert_called_once_with('1234')
        self.mockclient.show_metering_label.assert_not_called()

    def test_delete_failed(self):
        rsrc = self.create_metering_label()
        self.mockclient.delete_metering_label.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.label: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})
        self.mockclient.delete_metering_label.assert_called_once_with('1234')

    def test_attribute(self):
        rsrc = self.create_metering_label()
        self.mockclient.show_metering_label.return_value = {'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}}
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual('TestLabel', rsrc.FnGetAtt('name'))
        self.assertEqual('Description of TestLabel', rsrc.FnGetAtt('description'))
        self.assertTrue(rsrc.FnGetAtt('shared'))
        self.mockclient.create_metering_label.assert_called_once_with({'metering_label': {'name': 'TestLabel', 'description': 'Description of TestLabel', 'shared': True}})
        self.mockclient.show_metering_label.assert_called_with('1234')