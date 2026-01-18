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
class TestRBACEnforcerRest(_TestRBACEnforcerBase):

    def test_extract_subject_token_target_data(self):
        path = '/v3/auth/tokens'
        body = self._auth_json()
        with self.test_client() as c:
            r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
            token_id = r.headers['X-Subject-Token']
            c.get('/v3', headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
            token = PROVIDER_APIS.token_provider_api.validate_token(token_id)
            subj_token_data = self.enforcer._extract_subject_token_target_data()
            subj_token_data = subj_token_data['token']
            self.assertEqual(token.user_id, subj_token_data['user_id'])
            self.assertIn('user', subj_token_data)
            self.assertIn('domain', subj_token_data['user'])
            self.assertEqual(token.user_domain['id'], subj_token_data['user']['domain']['id'])

    def test_extract_filter_data(self):
        path = uuid.uuid4().hex

        @self.flask_blueprint.route('/%s' % path)
        def return_nothing_interesting():
            return ('OK', 200)
        self._register_blueprint_to_app()
        with self.test_client() as c:
            expected_param = uuid.uuid4().hex
            unexpected_param = uuid.uuid4().hex
            get_path = '/'.join([self.url_prefix, path])
            qs = '%(expected)s=EXPECTED&%(unexpected)s=UNEXPECTED' % {'expected': expected_param, 'unexpected': unexpected_param}
            c.get('%(path)s?%(qs)s' % {'path': get_path, 'qs': qs})
            extracted_filter = self.enforcer._extract_filter_values([expected_param])
            self.assertNotIn(extracted_filter, unexpected_param)
            self.assertIn(expected_param, expected_param)
            self.assertEqual(extracted_filter[expected_param], 'EXPECTED')

    def test_retrive_oslo_req_context(self):
        with self.test_client() as c:
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex))
            oslo_req_context = self.enforcer._get_oslo_req_context()
            self.assertIsInstance(oslo_req_context, context.RequestContext)

    def test_is_authenticated_check(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            self.enforcer._assert_is_authenticated()
            c.get('/', expected_status_code=300)
            self.assertRaises(exception.Unauthorized, self.enforcer._assert_is_authenticated)
            oslo_ctx = self.enforcer._get_oslo_req_context()
            oslo_ctx.authenticated = False
            self.assertRaises(exception.Unauthorized, self.enforcer._assert_is_authenticated)

    def test_extract_policy_check_credentials(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            extracted_creds = self.enforcer._extract_policy_check_credentials()
            self.assertEqual(flask.request.environ.get(authorization.AUTH_CONTEXT_ENV), extracted_creds)

    def test_extract_member_target_data_inferred(self):
        self.restful_api_resource.member_key = 'argument'
        member_from_driver = self._driver_simulation_get_method
        self.restful_api_resource.get_member_from_driver = member_from_driver
        argument_id = uuid.uuid4().hex
        with self.test_client() as c:
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id))
            extracted = self.enforcer._extract_member_target_data(member_target_type=None, member_target=None)
            self.assertDictEqual(extracted['target'], self.restful_api_resource().get(argument_id))

    def test_view_args_populated_in_policy_dict(self):

        def _enforce_mock_func(credentials, action, target, do_raise=True):
            if 'argument_id' not in target:
                raise exception.ForbiddenAction(action=action)
        self.useFixture(fixtures.MockPatchObject(self.enforcer, '_enforce', _enforce_mock_func))
        argument_id = uuid.uuid4().hex
        with self.test_client() as c:
            path = '/v3/auth/tokens'
            body = self._auth_json()
            r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
            token_id = r.headers['X-Subject-Token']
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id), headers={'X-Auth-Token': token_id})
            self.enforcer.enforce_call(action='example:allowed')
            c.get('%s/argument' % self.restful_api_url_prefix, headers={'X-Auth-Token': token_id})
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:allowed')

    def test_extract_member_target_data_supplied_target(self):
        member_type = uuid.uuid4().hex
        member_target = {uuid.uuid4().hex: {uuid.uuid4().hex}}
        extracted = self.enforcer._extract_member_target_data(member_target_type=member_type, member_target=member_target)
        self.assertDictEqual({'target': {member_type: member_target}}, extracted)

    def test_extract_member_target_data_bad_input(self):
        self.assertEqual({}, self.enforcer._extract_member_target_data(member_target=None, member_target_type=uuid.uuid4().hex))
        self.assertEqual({}, self.enforcer._extract_member_target_data(member_target={}, member_target_type=None))

    def test_call_build_enforcement_target(self):
        assertIn = self.assertIn
        assertEq = self.assertEqual
        ref_uuid = uuid.uuid4().hex

        def _enforce_mock_func(credentials, action, target, do_raise=True):
            assertIn('target.domain.id', target)
            assertEq(target['target.domain.id'], ref_uuid)

        def _build_enforcement_target():
            return {'domain': {'id': ref_uuid}}
        self.useFixture(fixtures.MockPatchObject(self.enforcer, '_enforce', _enforce_mock_func))
        argument_id = uuid.uuid4().hex
        with self.test_client() as c:
            path = '/v3/auth/tokens'
            body = self._auth_json()
            r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
            token_id = r.headers['X-Subject-Token']
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id), headers={'X-Auth-Token': token_id})
            self.enforcer.enforce_call(action='example:allowed', build_target=_build_enforcement_target)

    def test_policy_enforcer_action_decorator(self):
        action = 'example:allowed'

        @self.flask_blueprint.route('')
        @self.enforcer.policy_enforcer_action(action)
        def nothing_interesting():
            return ('OK', 200)
        self._register_blueprint_to_app()
        with self.test_client() as c:
            c.get('%s' % self.url_prefix)
            self.assertEqual(action, getattr(flask.g, self.enforcer.ACTION_STORE_ATTR))

    def test_policy_enforcer_action_invalid_action_decorator(self):

        def _decorator_fails():
            action = uuid.uuid4().hex

            @self.flask_blueprint.route('')
            @self.enforcer.policy_enforcer_action(action)
            def nothing_interesting():
                return ('OK', 200)
        self.assertRaises(ValueError, _decorator_fails)

    def test_enforce_call_invalid_action(self):
        self.assertRaises(exception.Forbidden, self.enforcer.enforce_call, action=uuid.uuid4().hex)

    def test_enforce_call_not_is_authenticated(self):
        with self.test_client() as c:
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex))
            with mock.patch.object(self.enforcer, '_get_oslo_req_context', return_value=None):
                self.assertRaises(exception.Unauthorized, self.enforcer.enforce_call, action='example:allowed')
            ctx = self.enforcer._get_oslo_req_context()
            ctx.authenticated = False
            self.assertRaises(exception.Unauthorized, self.enforcer.enforce_call, action='example:allowed')

    def test_enforce_call_explicit_target_attr(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
            target = {'myuser': {'id': self.user_req_admin['id']}}
            self.enforcer.enforce_call(action='example:target', target_attr=target)
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:subject_token', target_attr=target)

    def test_enforce_call_with_subject_token_data(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
            self.enforcer.enforce_call(action='example:subject_token')

    def test_enforce_call_with_member_target_type_and_member_target(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
            target_type = 'myuser'
            target = {'id': self.user_req_admin['id']}
            self.enforcer.enforce_call(action='example:target', member_target_type=target_type, member_target=target)
            self.enforcer.enforce_call(action='example:subject_token')

    def test_enforce_call_inferred_member_target_data(self):
        self.restful_api_resource.member_key = 'argument'
        member_from_driver = self._driver_simulation_get_method
        self.restful_api_resource.get_member_from_driver = member_from_driver
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
            self.enforcer.enforce_call(action='example:inferred_member_data')
            self.enforcer.enforce_call(action='example:subject_token')

    def test_enforce_call_with_filter_values(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s?user=%s' % (self.restful_api_url_prefix, uuid.uuid4().hex, self.user_req_admin['id']), headers={'X-Auth-Token': token_id})
            self.enforcer.enforce_call(action='example:with_filter', filters=['user'])
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:with_filter')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:with_filter', filters=['user'])
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:with_filter')

    def test_enforce_call_with_pre_instantiated_enforcer(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        enforcer = rbac_enforcer.enforcer.RBACEnforcer()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            self.enforcer.enforce_call(action='example:allowed', enforcer=enforcer)
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:denied', enforcer=enforcer)

    def test_enforce_call_sets_enforcement_attr(self):
        token_path = '/v3/auth/tokens'
        auth_json = self._auth_json()
        with self.test_client() as c:
            r = c.post(token_path, json=auth_json, expected_status_code=201)
            token_id = r.headers.get('X-Subject-Token')
            c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
            self.assertFalse(hasattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR))
            setattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR, False)
            self.enforcer.enforce_call(action='example:allowed')
            self.assertEqual(getattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR), True)
            setattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR, False)
            self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:denied')
            self.assertEqual(getattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR), True)