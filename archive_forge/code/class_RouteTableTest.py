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
class RouteTableTest(VPCTestBase):
    test_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_vpc:\n    Type: AWS::EC2::VPC\n    Properties:\n      CidrBlock: '10.0.0.0/16'\n  the_subnet:\n    Type: AWS::EC2::Subnet\n    Properties:\n      CidrBlock: 10.0.0.0/24\n      VpcId: {Ref: the_vpc}\n      AvailabilityZone: moon\n  the_route_table:\n    Type: AWS::EC2::RouteTable\n    Properties:\n      VpcId: {Ref: the_vpc}\n  the_association:\n    Type: AWS::EC2::SubnetRouteTableAssociation\n    Properties:\n      RouteTableId: {Ref: the_route_table}\n      SubnetId: {Ref: the_subnet}\n"

    def setUp(self):
        super(RouteTableTest, self).setUp()
        self.mock_create_network()
        self.mock_create_route_table()

    def test_route_table(self):
        stack = self.create_stack(self.test_template)
        self.mockclient.create_router.assert_called_with({'router': {'name': self.rt_name}})
        self.mockclient.add_interface_router.assert_has_calls([mock.call('bbbb', {'subnet_id': u'cccc'}), mock.call('ffff', {'subnet_id': u'cccc'})])
        route_table = stack['the_route_table']
        self.assertResourceState(route_table, 'ffff')
        self.mockclient.show_network.assert_called_with('aaaa')
        self.mockclient.create_network.assert_called_with({'network': {'name': self.vpc_name}})
        self.assertEqual(2, self.mockclient.create_router.call_count)
        association = stack['the_association']
        self.assertResourceState(association, 'the_association')
        scheduler.TaskRunner(association.delete)()
        scheduler.TaskRunner(route_table.delete)()
        stack.delete()
        self.mockclient.remove_interface_router.assert_has_calls([mock.call('ffff', {'subnet_id': u'cccc'}), mock.call('bbbb', {'subnet_id': u'cccc'})])
        self.mockclient.remove_gateway_router.assert_called_once_with('ffff')
        self.assertEqual(2, self.mockclient.show_subnet.call_count)
        self.mockclient.show_subnet.assert_called_with('cccc')
        self.assertEqual(2, self.mockclient.show_router.call_count)
        self.mockclient.show_router.assert_called_with('ffff')
        self.mockclient.delete_router.assert_called_once_with('ffff')