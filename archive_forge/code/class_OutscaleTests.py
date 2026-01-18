import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
class OutscaleTests(EC2Tests):

    def setUp(self):
        OutscaleSASNodeDriver.connectionCls.conn_class = EC2MockHttp
        EC2MockHttp.use_param = 'Action'
        EC2MockHttp.type = None
        self.driver = OutscaleSASNodeDriver(key=EC2_PARAMS[0], secret=EC2_PARAMS[1], host='some.outscalecloud.com')

    def test_ex_create_network(self):
        vpc = self.driver.ex_create_network('192.168.55.0/24', name='Test VPC')
        self.assertEqual('vpc-ad3527cf', vpc.id)
        self.assertEqual('192.168.55.0/24', vpc.cidr_block)
        self.assertEqual('pending', vpc.extra['state'])

    def test_ex_copy_image(self):
        image = self.driver.list_images()[0]
        try:
            self.driver.ex_copy_image('us-east-1', image, name='Faux Image', description='Test Image Copy')
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_ex_get_limits(self):
        try:
            self.driver.ex_get_limits()
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_ex_create_network_interface(self):
        subnet = self.driver.ex_list_subnets()[0]
        try:
            self.driver.ex_create_network_interface(subnet, name='Test Interface', description='My Test')
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_ex_delete_network_interface(self):
        interface = self.driver.ex_list_network_interfaces()[0]
        try:
            self.driver.ex_delete_network_interface(interface)
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_ex_attach_network_interface_to_node(self):
        node = self.driver.list_nodes()[0]
        interface = self.driver.ex_list_network_interfaces()[0]
        try:
            self.driver.ex_attach_network_interface_to_node(interface, node, 1)
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_ex_detach_network_interface(self):
        try:
            self.driver.ex_detach_network_interface('eni-attach-2b588b47')
        except NotImplementedError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        ids = [s.id for s in sizes]
        self.assertTrue('m1.small' in ids)
        self.assertTrue('m1.large' in ids)
        self.assertTrue('m1.xlarge' in ids)

    def test_ex_create_node_with_ex_iam_profile(self):
        image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
        size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
        self.assertRaises(NotImplementedError, self.driver.create_node, name='foo', image=image, size=size, ex_iamprofile='foo')

    def test_create_node_with_ex_userdata(self):
        EC2MockHttp.type = 'ex_user_data'
        image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
        size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
        result = self.driver.create_node(name='foo', image=image, size=size, ex_userdata='foo\nbar\x0coo')
        self.assertTrue(result)