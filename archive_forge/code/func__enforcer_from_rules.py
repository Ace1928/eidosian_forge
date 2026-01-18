import http.client as http
from unittest.mock import patch
from oslo_log.fixture import logging_error as log_fixture
from oslo_policy import policy
from oslo_utils.fixture import uuidsentinel as uuids
import testtools
import webob
import glance.api.middleware.cache
import glance.api.policy
from glance.common import exception
from glance import context
from glance.tests.unit import base
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import test_policy
from glance.tests.unit import utils as unit_test_utils
def _enforcer_from_rules(self, unparsed_rules):
    rules = policy.Rules.from_dict(unparsed_rules)
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    enforcer.set_rules(rules, overwrite=True)
    return enforcer