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
@_checks.register('field')
class FieldCheck(_checks.Check):
    """A non reversible check.

    All oslo.policy defined checks have a __str__ method with the property that
    rule == str(_parser.parse_rule(rule)). Consumers of oslo.policy may have
    defined checks for which that does not hold true. This FieldCheck is not
    reversible so we can use it for testing to ensure that this type of check
    does not break anything.
    """

    def __init__(self, kind, match):
        resource, field_value = match.split(':', 1)
        field, value = field_value.split('=', 1)
        super(FieldCheck, self).__init__(kind, '%s:%s:%s' % (resource, field, value))
        self.field = field
        self.value = value

    def __call__(self, target_dict, cred_dict, enforcer):
        return True