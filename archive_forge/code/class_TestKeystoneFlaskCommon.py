import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
class TestKeystoneFlaskCommon(rest.RestfulTestCase):
    _policy_rules = [policy.RuleDefault(name='example:allowed', check_str=''), policy.RuleDefault(name='example:deny', check_str='false:false')]

    def setUp(self):
        super(TestKeystoneFlaskCommon, self).setUp()
        enf = rbac_enforcer.enforcer.RBACEnforcer()

        def register_rules(enf_obj):
            enf_obj.register_defaults(self._policy_rules)
        self.useFixture(fixtures.MockPatchObject(enf, 'register_rules', register_rules))
        self.useFixture(fixtures.MockPatchObject(rbac_enforcer.enforcer, '_POSSIBLE_TARGET_ACTIONS', {r.name for r in self._policy_rules}))
        enf._reset()
        self.addCleanup(enf._reset)
        self.addCleanup(_TestResourceWithCollectionInfo._reset)

    def _get_token(self):
        auth_json = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.user_req_admin['name'], 'password': self.user_req_admin['password'], 'domain': {'id': self.user_req_admin['domain_id']}}}}, 'scope': {'project': {'id': self.project_service['id']}}}}
        return self.test_client().post('/v3/auth/tokens', json=auth_json, expected_status_code=201).headers['X-Subject-Token']

    def _setup_flask_restful_api(self, **options):
        self.restful_api_opts = options.copy()
        orig_value = _TestResourceWithCollectionInfo.api_prefix
        setattr(_TestResourceWithCollectionInfo, 'api_prefix', options.get('api_url_prefix', ''))
        self.addCleanup(setattr, _TestResourceWithCollectionInfo, 'api_prefix', orig_value)
        self.restful_api = _TestRestfulAPI(**options)
        self.public_app.app.register_blueprint(self.restful_api.blueprint)
        self.cleanup_instance('restful_api')
        self.cleanup_instance('restful_api_opts')

    def _make_requests(self):
        path_base = '/arguments'
        api_prefix = self.restful_api_opts.get('api_url_prefix', '')
        blueprint_prefix = self.restful_api._blueprint_url_prefix.rstrip('/')
        url = ''.join([x for x in [blueprint_prefix, api_prefix, path_base] if x])
        headers = {'X-Auth-Token': self._get_token()}
        with self.test_client() as c:
            resp = c.get(url, headers=headers)
            self.assertEqual(_TestResourceWithCollectionInfo.wrap_collection([]), resp.json)
            unknown_id = uuid.uuid4().hex
            c.get('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
            c.head('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
            c.put('%s/%s' % (url, unknown_id), json={}, headers=headers, expected_status_code=404)
            c.patch('%s/%s' % (url, unknown_id), json={}, headers=headers, expected_status_code=404)
            c.delete('%s/%s' % (url, unknown_id), headers=headers, expected_status_code=404)
            new_argument_resource = {'testing': uuid.uuid4().hex}
            new_argument_resp = c.post(url, json=new_argument_resource, headers=headers).json['argument']
            new_argument2_resource = {'testing': uuid.uuid4().hex}
            new_argument2_resp = c.post(url, json=new_argument2_resource, headers=headers).json['argument']
            get_list_resp = c.get(url, headers=headers).json
            self.assertIn(new_argument_resp, get_list_resp['arguments'])
            self.assertIn(new_argument2_resp, get_list_resp['arguments'])
            get_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
            self.assertEqual(new_argument_resp, get_resp)
            head_resp = c.head('%s/%s' % (url, new_argument_resp['id']), headers=headers).data
            self.assertEqual(head_resp, b'')
            replacement_argument = {'new_arg': True, 'id': uuid.uuid4().hex}
            c.put('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=replacement_argument, expected_status_code=400)
            replacement_argument.pop('id')
            c.put('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=replacement_argument)
            put_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
            self.assertNotIn(new_argument_resp['testing'], put_resp)
            self.assertTrue(put_resp['new_arg'])
            get_replacement_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
            self.assertEqual(put_resp, get_replacement_resp)
            patch_ref = {'uuid': uuid.uuid4().hex}
            patch_resp = c.patch('%s/%s' % (url, new_argument_resp['id']), headers=headers, json=patch_ref).json['argument']
            self.assertTrue(patch_resp['new_arg'])
            self.assertEqual(patch_ref['uuid'], patch_resp['uuid'])
            get_patched_ref_resp = c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers).json['argument']
            self.assertEqual(patch_resp, get_patched_ref_resp)
            c.delete('%s/%s' % (url, new_argument_resp['id']), headers=headers)
            c.get('%s/%s' % (url, new_argument_resp['id']), headers=headers, expected_status_code=404)

    def test_api_url_prefix(self):
        url_prefix = '/%s' % uuid.uuid4().hex
        self._setup_flask_restful_api(api_url_prefix=url_prefix)
        self._make_requests()

    def test_blueprint_url_prefix(self):
        url_prefix = '/%s' % uuid.uuid4().hex
        self._setup_flask_restful_api(blueprint_url_prefix=url_prefix)
        self._make_requests()

    def test_build_restful_api_no_prefix(self):
        self._setup_flask_restful_api()
        self._make_requests()

    def test_cannot_add_before_request_functions_twice(self):

        class TestAPIDuplicateBefore(_TestRestfulAPI):

            def __init__(self):
                super(TestAPIDuplicateBefore, self).__init__()
                self._register_before_request_functions()
        self.assertRaises(AssertionError, TestAPIDuplicateBefore)

    def test_cannot_add_after_request_functions_twice(self):

        class TestAPIDuplicateAfter(_TestRestfulAPI):

            def __init__(self):
                super(TestAPIDuplicateAfter, self).__init__()
                self._register_after_request_functions()
        self.assertRaises(AssertionError, TestAPIDuplicateAfter)

    def test_after_request_functions_must_be_added(self):

        class TestAPINoAfter(_TestRestfulAPI):

            def _register_after_request_functions(self, functions=None):
                pass
        self.assertRaises(AssertionError, TestAPINoAfter)

    def test_before_request_functions_must_be_added(self):

        class TestAPINoBefore(_TestRestfulAPI):

            def _register_before_request_functions(self, functions=None):
                pass
        self.assertRaises(AssertionError, TestAPINoBefore)

    def test_before_request_functions(self):
        attr = uuid.uuid4().hex

        def do_something():
            setattr(flask.g, attr, True)

        class TestAPI(_TestRestfulAPI):

            def _register_before_request_functions(self, functions=None):
                functions = functions or []
                functions.append(do_something)
                super(TestAPI, self)._register_before_request_functions(functions)
        api = TestAPI(resources=[_TestResourceWithCollectionInfo])
        self.public_app.app.register_blueprint(api.blueprint)
        token = self._get_token()
        with self.test_client() as c:
            c.get('/v3/arguments', headers={'X-Auth-Token': token})
            self.assertTrue(getattr(flask.g, attr, False))

    def test_after_request_functions(self):
        attr = uuid.uuid4().hex

        def do_something(resp):
            setattr(flask.g, attr, True)
            resp.status_code = 420
            return resp

        class TestAPI(_TestRestfulAPI):

            def _register_after_request_functions(self, functions=None):
                functions = functions or []
                functions.append(do_something)
                super(TestAPI, self)._register_after_request_functions(functions)
        api = TestAPI(resources=[_TestResourceWithCollectionInfo])
        self.public_app.app.register_blueprint(api.blueprint)
        token = self._get_token()
        with self.test_client() as c:
            c.get('/v3/arguments', headers={'X-Auth-Token': token}, expected_status_code=420)

    def test_construct_resource_map(self):
        resource_name = 'arguments'
        param_relation = json_home.build_v3_parameter_relation('argument_id')
        alt_rel_func = functools.partial(json_home.build_v3_extension_resource_relation, extension_name='extension', extension_version='1.0')
        url = '/v3/arguments/<string:argument_id>'
        old_url = [dict(url='/v3/old_arguments/<string:argument_id>', json_home=flask_common.construct_json_home_data(rel='arguments', resource_relation_func=alt_rel_func))]
        mapping = flask_common.construct_resource_map(resource=_TestResourceWithCollectionInfo, url=url, resource_kwargs={}, alternate_urls=old_url, rel=resource_name, status=json_home.Status.EXPERIMENTAL, path_vars={'argument_id': param_relation}, resource_relation_func=json_home.build_v3_resource_relation)
        self.assertEqual(_TestResourceWithCollectionInfo, mapping.resource)
        self.assertEqual(url, mapping.url)
        self.assertEqual(json_home.build_v3_resource_relation(resource_name), mapping.json_home_data.rel)
        self.assertEqual(json_home.Status.EXPERIMENTAL, mapping.json_home_data.status)
        self.assertEqual({'argument_id': param_relation}, mapping.json_home_data.path_vars)
        self.assertEqual(1, len(mapping.alternate_urls))
        alt_url_data = mapping.alternate_urls[0]
        self.assertEqual(old_url[0]['url'], alt_url_data['url'])
        self.assertEqual(old_url[0]['json_home'], alt_url_data['json_home'])

    def test_instantiate_and_register_to_app(self):
        self.restful_api_opts = {}
        self.restful_api = _TestRestfulAPI.instantiate_and_register_to_app(self.public_app.app)
        self.cleanup_instance('restful_api_opts')
        self.cleanup_instance('restful_api')
        self._make_requests()

    def test_unenforced_api_decorator(self):

        class MappedResource(flask_restful.Resource):

            @flask_common.unenforced_api
            def post(self):
                post_body = flask.request.get_json()
                return ({'post_body': post_body}, 201)
        resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
        restful_api = _TestRestfulAPI(resource_mapping=[resource_map], resources=[])
        self.public_app.app.register_blueprint(restful_api.blueprint)
        token = self._get_token()
        with self.test_client() as c:
            body = {'test_value': uuid.uuid4().hex}
            resp = c.post('/v3/test_api', json=body, headers={'X-Auth-Token': token})
            self.assertEqual(body, resp.json['post_body'])
            resp = c.post('/v3/test_api', json=body)
            self.assertEqual(body, resp.json['post_body'])

    def test_HTTP_OPTIONS_is_unenforced(self):

        class MappedResource(flask_restful.Resource):

            def post(self):
                pass
        resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
        restful_api = _TestRestfulAPI(resource_mapping=[resource_map], resources=[])
        self.public_app.app.register_blueprint(restful_api.blueprint)
        with self.test_client() as c:
            r = c.options('/v3/test_api')
            self.assertEqual(set(['OPTIONS', 'POST']), set([v.lstrip().rstrip() for v in r.headers['Allow'].split(',')]))
            self.assertEqual(r.headers['Content-Length'], '0')
            self.assertEqual(r.data, b'')

    def test_mapped_resource_routes(self):

        class MappedResource(flask_restful.Resource):

            def post(self):
                rbac_enforcer.enforcer.RBACEnforcer().enforce_call(action='example:allowed')
                post_body = flask.request.get_json()
                return ({'post_body': post_body}, 201)
        resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
        restful_api = _TestRestfulAPI(resource_mapping=[resource_map], resources=[])
        self.public_app.app.register_blueprint(restful_api.blueprint)
        token = self._get_token()
        with self.test_client() as c:
            body = {'test_value': uuid.uuid4().hex}
            resp = c.post('/v3/test_api', json=body, headers={'X-Auth-Token': token})
            self.assertEqual(body, resp.json['post_body'])

    def test_correct_json_home_document(self):

        class MappedResource(flask_restful.Resource):

            def post(self):
                rbac_enforcer.enforcer.RBACEnforcer().enforce_call(action='example:allowed')
                post_body = flask.request.get_json()
                return {'post_body': post_body}
        json_home_data = {'https://docs.openstack.org/api/openstack-identity/3/rel/argument': {'href-template': '/v3/arguments/{argument_id}', 'href-vars': {'argument_id': 'https://docs.openstack.org/api/openstack-identity/3/param/argument_id'}}, 'https://docs.openstack.org/api/openstack-identity/3/rel/arguments': {'href': '/v3/arguments'}, 'https://docs.openstack.org/api/openstack-identity/3/rel/test': {'href': '/v3/test_api'}}
        resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
        restful_api = _TestRestfulAPI(resource_mapping=[resource_map])
        self.public_app.app.register_blueprint(restful_api.blueprint)
        with self.test_client() as c:
            headers = {'Accept': 'application/json-home'}
            resp = c.get('/', headers=headers)
            resp_data = jsonutils.loads(resp.data)
            for rel in json_home_data:
                self.assertThat(resp_data['resources'][rel], matchers.Equals(json_home_data[rel]))

    def test_normalize_domain_id_extracts_domain_id_if_needed(self):
        self._setup_flask_restful_api()
        blueprint_prefix = self.restful_api._blueprint_url_prefix.rstrip('/')
        url = ''.join([blueprint_prefix, '/arguments'])
        headers = {'X-Auth-Token': self._get_token()}
        ref_with_domain_id = {'domain_id': uuid.uuid4().hex}
        ref_without_domain_id = {}
        with self.test_client() as c:
            c.get('%s/%s' % (url, uuid.uuid4().hex), headers=headers, expected_status_code=404)
            oslo_context = flask.request.environ[context.REQUEST_CONTEXT_ENV]
            domain_id = ref_with_domain_id['domain_id']
            flask_common.ResourceBase._normalize_domain_id(ref_with_domain_id)
            self.assertEqual(domain_id, ref_with_domain_id['domain_id'])
            flask_common.ResourceBase._normalize_domain_id(ref_without_domain_id)
            self.assertEqual(CONF.identity.default_domain_id, ref_without_domain_id['domain_id'])
            ref_without_domain_id.clear()
            oslo_context.domain_id = uuid.uuid4().hex
            flask_common.ResourceBase._normalize_domain_id(ref_with_domain_id)
            self.assertEqual(domain_id, ref_with_domain_id['domain_id'])
            flask_common.ResourceBase._normalize_domain_id(ref_without_domain_id)
            self.assertEqual(oslo_context.domain_id, ref_without_domain_id['domain_id'])
            ref_without_domain_id.clear()
            oslo_context.is_admin = True
            oslo_context.domain_id = None
            flask_common.ResourceBase._normalize_domain_id(ref_with_domain_id)
            self.assertEqual(domain_id, ref_with_domain_id['domain_id'])
            self.assertRaises(exception.ValidationError, flask_common.ResourceBase._normalize_domain_id, ref=ref_without_domain_id)

    def test_api_prefix_self_referential_link_substitution(self):
        view_arg = uuid.uuid4().hex

        class TestResource(flask_common.ResourceBase):
            api_prefix = '/<string:test_value>/nothing'
        with self.test_request_context(path='/%s/nothing/values' % view_arg, base_url='https://localhost/'):
            flask.request.view_args = {'test_value': view_arg}
            ref = {'id': uuid.uuid4().hex}
            TestResource._add_self_referential_link(ref, collection_name='values')
            self.assertTrue(ref['links']['self'].startswith('https://localhost/v3/%s' % view_arg))

    def test_json_body_before_req_func_valid_json(self):
        with self.test_request_context(headers={'Content-Type': 'application/json'}, data='{"key": "value"}'):
            json_body.json_body_before_request()

    def test_json_body_before_req_func_invalid_json(self):
        with self.test_request_context(headers={'Content-Type': 'application/json'}, data='invalid JSON'):
            self.assertRaises(exception.ValidationError, json_body.json_body_before_request)

    def test_json_body_before_req_func_no_content_type(self):
        with self.test_request_context(data='{"key": "value"}'):
            json_body.json_body_before_request()
        with self.test_request_context(headers={'Content-Type': ''}, data='{"key": "value"}'):
            json_body.json_body_before_request()

    def test_json_body_before_req_func_unrecognized_content_type(self):
        with self.test_request_context(headers={'Content-Type': 'unrecognized/content-type'}, data='{"key": "value"'):
            self.assertRaises(exception.ValidationError, json_body.json_body_before_request)

    def test_json_body_before_req_func_unrecognized_conten_type_no_body(self):
        with self.test_request_context(headers={'Content-Type': 'unrecognized/content-type'}):
            json_body.json_body_before_request()

    def test_resource_collection_key_raises_exception_if_unset(self):

        class TestResource(flask_common.ResourceBase):
            """A Test Resource."""

        class TestResourceWithKey(flask_common.ResourceBase):
            collection_key = uuid.uuid4().hex
        r = TestResource()
        self.assertRaises(ValueError, getattr, r, 'collection_key')
        r = TestResourceWithKey()
        self.assertEqual(TestResourceWithKey.collection_key, r.collection_key)

    def test_resource_member_key_raises_exception_if_unset(self):

        class TestResource(flask_common.ResourceBase):
            """A Test Resource."""

        class TestResourceWithKey(flask_common.ResourceBase):
            member_key = uuid.uuid4().hex
        r = TestResource()
        self.assertRaises(ValueError, getattr, r, 'member_key')
        r = TestResourceWithKey()
        self.assertEqual(TestResourceWithKey.member_key, r.member_key)