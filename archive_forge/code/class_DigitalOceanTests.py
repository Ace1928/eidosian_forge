import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
class DigitalOceanTests(LibcloudTestCase):

    def setUp(self):
        DigitalOceanBaseDriver.connectionCls.conn_class = DigitalOceanMockHttp
        DigitalOceanMockHttp.type = None
        self.driver = DigitalOceanBaseDriver(*DIGITALOCEAN_v2_PARAMS)

    def test_authentication(self):
        DigitalOceanMockHttp.type = 'UNAUTHORIZED'
        self.assertRaises(InvalidCredsError, self.driver.ex_account_info)

    def test_ex_account_info(self):
        account_info = self.driver.ex_account_info()
        self.assertEqual(account_info['uuid'], 'a1234567890b1234567890c1234567890d12345')
        self.assertTrue(account_info['email_verified'])
        self.assertEqual(account_info['email'], 'user@domain.tld')
        self.assertEqual(account_info['droplet_limit'], 10)

    def test_ex_list_events(self):
        events = self.driver.ex_list_events()
        self.assertEqual(events, [])

    def test_ex_get_event(self):
        action = self.driver.ex_get_event('12345670')
        self.assertEqual(action['id'], 12345670)
        self.assertEqual(action['status'], 'completed')
        self.assertEqual(action['type'], 'power_on')

    def test__paginated_request(self):
        DigitalOceanMockHttp.type = 'page_1'
        actions = self.driver._paginated_request('/v2/actions', 'actions')
        self.assertEqual(actions[0]['id'], 12345671)
        self.assertEqual(actions[0]['status'], 'completed')