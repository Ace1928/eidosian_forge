from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class RoleCheckTestCase(base.PolicyBaseTestCase):

    def test_accept(self):
        check = _checks.RoleCheck('role', 'sPaM')
        self.assertTrue(check({}, dict(roles=['SpAm']), self.enforcer))

    def test_reject(self):
        check = _checks.RoleCheck('role', 'spam')
        self.assertFalse(check({}, dict(roles=[]), self.enforcer))

    def test_format_value(self):
        check = _checks.RoleCheck('role', '%(target.role.name)s')
        target_dict = {'target.role.name': 'a'}
        cred_dict = dict(user='user', roles=['a', 'b', 'c'])
        self.assertTrue(check(target_dict, cred_dict, self.enforcer))
        target_dict = {'target.role.name': 'd'}
        self.assertFalse(check(target_dict, cred_dict, self.enforcer))
        target_dict = dict(target=dict(role=dict()))
        self.assertFalse(check(target_dict, cred_dict, self.enforcer))

    def test_no_roles_case(self):
        check = _checks.RoleCheck('role', 'spam')
        self.assertFalse(check({}, {}, self.enforcer))