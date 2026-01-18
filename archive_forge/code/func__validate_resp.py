import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
def _validate_resp(self, resp, status_code):
    self.assertEqual(status_code, resp.status_code)
    self.assertEqual('https://placement.example.com/allocation_candidates', resp.url)
    self.assert_calls()