import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _assert_expected_implied_role_response(self, expected_prior_id, expected_implied_ids):
    r = self.get('/roles/%s/implies' % expected_prior_id)
    response = r.json
    role_inference = response['role_inference']
    self.assertEqual(expected_prior_id, role_inference['prior_role']['id'])
    prior_link = '/v3/roles/' + expected_prior_id + '/implies'
    self.assertThat(response['links']['self'], matchers.EndsWith(prior_link))
    actual_implied_ids = [implied['id'] for implied in role_inference['implies']]
    self.assertCountEqual(expected_implied_ids, actual_implied_ids)
    self.assertIsNotNone(role_inference['prior_role']['links']['self'])
    for implied in role_inference['implies']:
        self.assertIsNotNone(implied['links']['self'])