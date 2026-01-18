from tests.compat import mock
import re
import xml.dom.minidom
from boto.exception import BotoServerError
from boto.route53.connection import Route53Connection
from boto.route53.exception import DNSServerError
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets, Record
from boto.route53.zone import Zone
from nose.plugins.attrib import attr
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
@attr(route53=True)
class TestGetCheckerIpRanges(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def default_body(self):
        return b'\n<GetCheckerIpRangesResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n   <CheckerIpRanges>\n      <member>54.183.255.128/26</member>\n      <member>54.228.16.0/26</member>\n      <member>54.232.40.64/26</member>\n      <member>177.71.207.128/26</member>\n      <member>176.34.159.192/26</member>\n   </CheckerIpRanges>\n</GetCheckerIpRangesResponse>\n        '

    def test_get_checker_ip_ranges(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_checker_ip_ranges()
        ip_ranges = response['GetCheckerIpRangesResponse']['CheckerIpRanges']
        self.assertEqual(len(ip_ranges), 5)
        self.assertIn('54.183.255.128/26', ip_ranges)
        self.assertIn('54.228.16.0/26', ip_ranges)
        self.assertIn('54.232.40.64/26', ip_ranges)
        self.assertIn('177.71.207.128/26', ip_ranges)
        self.assertIn('176.34.159.192/26', ip_ranges)