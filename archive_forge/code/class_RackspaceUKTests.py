import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node
from libcloud.test.secrets import DNS_PARAMS_RACKSPACE
from libcloud.loadbalancer.base import LoadBalancer
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rackspace import RackspaceDNSDriver, RackspacePTRRecord
class RackspaceUKTests(RackspaceUSTests):
    klass = RackspaceDNSDriver
    endpoint_url = 'https://lon.dns.api.rackspacecloud.com/v1.0/11111'
    region = 'uk'