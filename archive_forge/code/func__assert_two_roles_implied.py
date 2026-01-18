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
def _assert_two_roles_implied(self):
    self._assert_expected_implied_role_response(self.prior['id'], [self.implied1['id'], self.implied2['id']])
    self._assert_expected_role_inference_rule_response(self.prior['id'], self.implied1['id'])
    self._assert_expected_role_inference_rule_response(self.prior['id'], self.implied2['id'])