from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
class _TestRBACEnforcerBase(rest.RestfulTestCase):

    def setUp(self):
        super(_TestRBACEnforcerBase, self).setUp()
        self._setup_enforcer_object()
        self._setup_dynamic_flask_blueprint_api()
        self._setup_flask_restful_api()

    def _setup_enforcer_object(self):
        self.enforcer = rbac_enforcer.enforcer.RBACEnforcer()
        self.cleanup_instance('enforcer')

        def register_new_rules(enforcer):
            rules = self._testing_policy_rules()
            enforcer.register_defaults(rules)
        self.useFixture(fixtures.MockPatchObject(self.enforcer, 'register_rules', register_new_rules))
        original_actions = rbac_enforcer.enforcer._POSSIBLE_TARGET_ACTIONS
        rbac_enforcer.enforcer._POSSIBLE_TARGET_ACTIONS = frozenset([rule.name for rule in self._testing_policy_rules()])
        self.addCleanup(setattr, rbac_enforcer.enforcer, '_POSSIBLE_TARGET_ACTIONS', original_actions)
        self.enforcer._reset()

    def _setup_dynamic_flask_blueprint_api(self):
        api = uuid.uuid4().hex
        url_prefix = '/_%s_TEST' % api
        blueprint = blueprints.Blueprint(api, __name__, url_prefix=url_prefix)
        self.url_prefix = url_prefix
        self.flask_blueprint = blueprint
        self.cleanup_instance('flask_blueprint', 'url_prefix')

    def _driver_simulation_get_method(self, argument_id):
        user = self.user_req_admin
        return {'id': argument_id, 'value': 'TEST', 'owner_id': user['id']}

    def _setup_flask_restful_api(self):
        self.restful_api_url_prefix = '/_%s_TEST' % uuid.uuid4().hex
        self.restful_api = flask_restful.Api(self.public_app.app, self.restful_api_url_prefix)
        driver_simulation_method = self._driver_simulation_get_method

        class RestfulResource(flask_restful.Resource):

            def get(self, argument_id=None):
                if argument_id is not None:
                    return self._get_argument(argument_id)
                return self._list_arguments()

            def _get_argument(self, argument_id):
                return {'argument': driver_simulation_method(argument_id)}

            def _list_arguments(self):
                return {'arguments': []}
        self.restful_api_resource = RestfulResource
        self.restful_api.add_resource(RestfulResource, '/argument/<string:argument_id>', '/argument')
        self.cleanup_instance('restful_api', 'restful_resource', 'restful_api_url_prefix')

    def _register_blueprint_to_app(self):
        self.public_app.app.register_blueprint(self.flask_blueprint, url_prefix=self.url_prefix)

    def _auth_json(self):
        return {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.user_req_admin['name'], 'password': self.user_req_admin['password'], 'domain': {'id': self.user_req_admin['domain_id']}}}}, 'scope': {'project': {'id': self.project_service['id']}}}}

    def _testing_policy_rules(self):
        test_policy_rules = [policy.RuleDefault(name='example:subject_token', check_str='user_id:%(target.token.user_id)s', scope_types=['project']), policy.RuleDefault(name='example:target', check_str='user_id:%(target.myuser.id)s', scope_types=['project']), policy.RuleDefault(name='example:inferred_member_data', check_str='user_id:%(target.argument.owner_id)s', scope_types=['project']), policy.RuleDefault(name='example:with_filter', check_str='user_id:%(user)s', scope_types=['project']), policy.RuleDefault(name='example:allowed', check_str='', scope_types=['project']), policy.RuleDefault(name='example:denied', check_str='false:false', scope_types=['project'])]
        return test_policy_rules