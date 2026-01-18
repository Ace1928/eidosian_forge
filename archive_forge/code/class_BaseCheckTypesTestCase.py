import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
class BaseCheckTypesTestCase(base.PolicyBaseTestCase):

    @mock.patch.object(_checks, 'registered_checks', {})
    def test_base_check_types_are_public(self):
        """Check that those check types are part of public API.

           They are blessed to be used by library consumers.
        """
        for check_type in (policy.AndCheck, policy.NotCheck, policy.OrCheck, policy.RuleCheck):

            class TestCheck(check_type):
                pass
            check_str = str(check_type)
            policy.register(check_str, TestCheck)
            self.assertEqual(TestCheck, _checks.registered_checks[check_str], message='%s check type is not public.' % check_str)