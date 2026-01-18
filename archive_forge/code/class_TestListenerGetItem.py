import xml.sax
from tests.unit import unittest
import boto.resultset
from boto.ec2.elb.loadbalancer import LoadBalancer
from boto.ec2.elb.listener import Listener
class TestListenerGetItem(unittest.TestCase):

    def test_getitem_for_http_listener(self):
        listener = Listener(load_balancer_port=80, instance_port=80, protocol='HTTP', instance_protocol='HTTP')
        self.assertEqual(listener[0], 80)
        self.assertEqual(listener[1], 80)
        self.assertEqual(listener[2], 'HTTP')
        self.assertEqual(listener[3], 'HTTP')

    def test_getitem_for_https_listener(self):
        listener = Listener(load_balancer_port=443, instance_port=80, protocol='HTTPS', instance_protocol='HTTP', ssl_certificate_id='look_at_me_im_an_arn')
        self.assertEqual(listener[0], 443)
        self.assertEqual(listener[1], 80)
        self.assertEqual(listener[2], 'HTTPS')
        self.assertEqual(listener[3], 'HTTP')
        self.assertEqual(listener[4], 'look_at_me_im_an_arn')