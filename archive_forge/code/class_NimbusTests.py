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
class NimbusTests(EC2Tests):

    def setUp(self):
        NimbusNodeDriver.connectionCls.conn_class = EC2MockHttp
        EC2MockHttp.use_param = 'Action'
        EC2MockHttp.type = None
        self.driver = NimbusNodeDriver(key=EC2_PARAMS[0], secret=EC2_PARAMS[1], host='some.nimbuscloud.com')

    def test_ex_describe_addresses_for_node(self):
        node = Node('i-4382922a', None, None, None, None, self.driver)
        ip_addresses = self.driver.ex_describe_addresses_for_node(node)
        self.assertEqual(len(ip_addresses), 0)

    def test_ex_describe_addresses(self):
        node = Node('i-4382922a', None, None, None, None, self.driver)
        nodes_elastic_ips = self.driver.ex_describe_addresses([node])
        self.assertEqual(len(nodes_elastic_ips), 1)
        self.assertEqual(len(nodes_elastic_ips[node.id]), 0)

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        ids = [s.id for s in sizes]
        self.assertTrue('m1.small' in ids)
        self.assertTrue('m1.large' in ids)
        self.assertTrue('m1.xlarge' in ids)

    def test_list_nodes(self):
        node = self.driver.list_nodes()[0]
        self.assertExecutedMethodCount(0)
        public_ips = node.public_ips
        self.assertEqual(node.id, 'i-4382922a')
        self.assertEqual(len(node.public_ips), 1)
        self.assertEqual(public_ips[0], '1.2.3.4')
        self.assertEqual(node.extra['tags'], {})
        node = self.driver.list_nodes()[1]
        self.assertExecutedMethodCount(0)
        public_ips = node.public_ips
        self.assertEqual(node.id, 'i-8474834a')
        self.assertEqual(len(node.public_ips), 1)
        self.assertEqual(public_ips[0], '1.2.3.5')
        self.assertEqual(node.extra['tags'], {'Name': 'Test Server 2', 'Group': 'VPC Test'})

    def test_ex_create_tags(self):
        node = self.driver.list_nodes()[0]
        self.driver.ex_create_tags(resource=node, tags={'foo': 'bar'})
        self.assertExecutedMethodCount(0)

    def test_create_volume_snapshot_with_tags(self):
        vol = StorageVolume(id='vol-4282672b', name='test', state=StorageVolumeState.AVAILABLE, size=10, driver=self.driver)
        snap = self.driver.create_volume_snapshot(vol, 'Test snapshot', ex_metadata={'my_tag': 'test'})
        self.assertDictEqual({}, snap.extra['tags'])