from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SubnetTest(VPCTestBase):
    test_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_vpc:\n    Type: AWS::EC2::VPC\n    Properties: {CidrBlock: '10.0.0.0/16'}\n  the_subnet:\n    Type: AWS::EC2::Subnet\n    Properties:\n      CidrBlock: 10.0.0.0/24\n      VpcId: {Ref: the_vpc}\n      AvailabilityZone: moon\n"

    def test_subnet(self):
        self.mock_create_network()
        exc = neutron_exc.NeutronClientException(status_code=404)
        self.mockclient.remove_interface_router.side_effect = exc
        self.mockclient.delete_subnet.side_effect = exc
        stack = self.create_stack(self.test_template)
        subnet = stack['the_subnet']
        self.assertResourceState(subnet, 'cccc')
        self.mockclient.list_routers.assert_called_with(name=self.vpc_name)
        self.validate_mock_create_network()
        self.mockclient.add_interface_router.assert_called_once_with(u'bbbb', {'subnet_id': 'cccc'})
        self.mockclient.create_subnet.assert_called_once_with({'subnet': {'network_id': u'aaaa', 'cidr': u'10.0.0.0/24', 'ip_version': 4, 'name': self.subnet_name}})
        self.assertEqual(4, self.mockclient.show_network.call_count)
        self.assertRaises(exception.InvalidTemplateAttribute, subnet.FnGetAtt, 'Foo')
        self.assertEqual('moon', subnet.FnGetAtt('AvailabilityZone'))
        scheduler.TaskRunner(subnet.delete)()
        subnet.state_set(subnet.CREATE, subnet.COMPLETE, 'to delete again')
        scheduler.TaskRunner(subnet.delete)()
        scheduler.TaskRunner(stack['the_vpc'].delete)()
        self.mockclient.show_network.assert_called_with('aaaa')
        self.mockclient.list_routers.assert_called_with(name=self.vpc_name)
        self.assertEqual(5, self.mockclient.list_routers.call_count)
        self.assertEqual(7, self.mockclient.show_network.call_count)

    def _mock_create_subnet_failed(self, stack_name):
        self.subnet_name = utils.PhysName(stack_name, 'the_subnet')
        self.mockclient.create_subnet.return_value = {'subnet': {'status': 'ACTIVE', 'name': self.subnet_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'cccc'}}
        exc = neutron_exc.NeutronClientException(status_code=404)
        self.mockclient.show_network.side_effect = exc

    def test_create_failed_delete_success(self):
        stack_name = 'test_subnet_'
        self._mock_create_subnet_failed(stack_name)
        t = template_format.parse(self.test_template)
        tmpl = template.Template(t)
        stack = parser.Stack(utils.dummy_context(), stack_name, tmpl, stack_id=str(uuid.uuid4()))
        tmpl.t['Resources']['the_subnet']['Properties']['VpcId'] = 'aaaa'
        resource_defns = tmpl.resource_definitions(stack)
        rsrc = sn.Subnet('the_subnet', resource_defns['the_subnet'], stack)
        rsrc.validate()
        self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        ref_id = rsrc.FnGetRefId()
        self.assertEqual(u'cccc', ref_id)
        self.mockclient.create_subnet.assert_called_once_with({'subnet': {'network_id': u'aaaa', 'cidr': u'10.0.0.0/24', 'ip_version': 4, 'name': self.subnet_name}})
        self.assertEqual(1, self.mockclient.show_network.call_count)
        self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(2, self.mockclient.show_network.call_count)
        self.mockclient.delete_subnet.assert_called_once_with('cccc')