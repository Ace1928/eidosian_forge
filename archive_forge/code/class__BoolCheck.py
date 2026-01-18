from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class _BoolCheck(_checks.BaseCheck):

    def __init__(self, result):
        self.called = False
        self.result = result

    def __str__(self):
        return str(self.result)

    def __call__(self, target, creds, enforcer, current_rule=None):
        self.called = True
        return self.result