from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class TestImpliedRoles(base.V3ClientTestCase):

    def setUp(self):
        super(TestImpliedRoles, self).setUp()

    def test_implied_roles(self):
        initial_rule_count = len(self.client.inference_rules.list_inference_roles())
        self.create_roles()
        self.create_rules()
        rule_count = len(self.client.inference_rules.list_inference_roles())
        self.assertEqual(initial_rule_count + len(inference_rules), rule_count)

    def role_dict(self):
        roles = {role.name: role.id for role in self.client.roles.list()}
        return roles

    def create_roles(self):
        for role_def in role_defs:
            role = fixtures.Role(self.client, name=role_def)
            self.useFixture(role)

    def create_rules(self):
        roles = self.role_dict()
        for prior, implied in inference_rules.items():
            rule = fixtures.InferenceRule(self.client, roles[prior], roles[implied])
            self.useFixture(rule)