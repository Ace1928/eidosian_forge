import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
class RackspaceUKLBTests(RackspaceLBTests):

    def setUp(self):
        RackspaceLBDriver.connectionCls.conn_class = RackspaceLBMockHttp
        RackspaceLBMockHttp.type = None
        self.driver = RackspaceLBDriver('user', 'key', region='lon')
        self.driver.connection._populate_hosts_and_request_paths()