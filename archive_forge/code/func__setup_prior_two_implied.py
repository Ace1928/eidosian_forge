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
def _setup_prior_two_implied(self):
    self.prior = self._create_role()
    self.implied1 = self._create_role()
    self._create_implied_role(self.prior, self.implied1)
    self.implied2 = self._create_role()
    self._create_implied_role(self.prior, self.implied2)