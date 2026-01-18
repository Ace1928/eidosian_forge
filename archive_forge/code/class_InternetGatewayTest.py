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
class InternetGatewayTest(VPCTestBase):
    test_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_gateway:\n    Type: AWS::EC2::InternetGateway\n  the_vpc:\n    Type: AWS::EC2::VPC\n    Properties:\n      CidrBlock: '10.0.0.0/16'\n  the_subnet:\n    Type: AWS::EC2::Subnet\n    Properties:\n      CidrBlock: 10.0.0.0/24\n      VpcId: {Ref: the_vpc}\n      AvailabilityZone: moon\n  the_attachment:\n    Type: AWS::EC2::VPCGatewayAttachment\n    Properties:\n      VpcId: {Ref: the_vpc}\n      InternetGatewayId: {Ref: the_gateway}\n  the_route_table:\n    Type: AWS::EC2::RouteTable\n    Properties:\n      VpcId: {Ref: the_vpc}\n  the_association:\n    Type: AWS::EC2::SubnetRouteTableAssociation\n    Properties:\n      RouteTableId: {Ref: the_route_table}\n      SubnetId: {Ref: the_subnet}\n"

    def setUp(self):
        super(InternetGatewayTest, self).setUp()
        self.mock_create_internet_gateway()
        self.mock_create_network()
        self.mock_create_route_table()

    def mock_create_internet_gateway(self):
        self.mockclient.list_networks.return_value = {'networks': [{'status': 'ACTIVE', 'subnets': [], 'name': 'nova', 'router:external': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'admin_state_up': True, 'shared': True, 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}]}

    def test_internet_gateway(self):
        stack = self.create_stack(self.test_template)
        self.mockclient.show_network.assert_called_with('aaaa')
        self.mockclient.create_network.assert_called_with({'network': {'name': self.vpc_name}})
        self.assertEqual(2, self.mockclient.create_router.call_count)
        self.mockclient.create_router.assert_called_with({'router': {'name': self.rt_name}})
        self.mockclient.add_interface_router.assert_has_calls([mock.call('bbbb', {'subnet_id': u'cccc'}), mock.call('ffff', {'subnet_id': u'cccc'})])
        self.mockclient.list_networks.assert_called_once_with(**{'router:external': True})
        gateway = stack['the_gateway']
        self.assertResourceState(gateway, gateway.physical_resource_name())
        self.mockclient.add_gateway_router.assert_called_with('ffff', {'network_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'})
        attachment = stack['the_attachment']
        self.assertResourceState(attachment, 'the_attachment')
        route_table = stack['the_route_table']
        self.assertEqual([route_table], list(attachment._vpc_route_tables()))
        stack.delete()
        self.mockclient.remove_interface_router.assert_has_calls([mock.call('ffff', {'subnet_id': u'cccc'}), mock.call('bbbb', {'subnet_id': u'cccc'})])
        self.mockclient.remove_gateway_router.assert_called_with('ffff')
        self.assertEqual(2, self.mockclient.remove_gateway_router.call_count)
        self.assertEqual(2, self.mockclient.show_subnet.call_count)
        self.mockclient.show_subnet.assert_called_with('cccc')
        self.mockclient.show_router.assert_called_with('ffff')
        self.assertEqual(2, self.mockclient.show_router.call_count)
        self.mockclient.delete_router.assert_called_once_with('ffff')