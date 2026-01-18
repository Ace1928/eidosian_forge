import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
class InstantiationDataTest(common.HeatTestCase):

    def test_parse_error_success(self):
        with stacks.InstantiationData.parse_error_check('Garbage'):
            pass

    def test_parse_error(self):

        def generate_error():
            with stacks.InstantiationData.parse_error_check('Garbage'):
                raise ValueError
        self.assertRaises(webob.exc.HTTPBadRequest, generate_error)

    def test_parse_error_message(self):
        bad_temp = "\nheat_template_version: '2013-05-23'\nparameters:\n  KeyName:\n     type: string\n    description: bla\n        "

        def generate_error():
            with stacks.InstantiationData.parse_error_check('foo'):
                template_format.parse(bad_temp)
        parse_ex = self.assertRaises(webob.exc.HTTPBadRequest, generate_error)
        self.assertIn('foo', str(parse_ex))

    def test_stack_name(self):
        body = {'stack_name': 'wibble'}
        data = stacks.InstantiationData(body)
        self.assertEqual('wibble', data.stack_name())

    def test_stack_name_missing(self):
        body = {'not the stack_name': 'wibble'}
        data = stacks.InstantiationData(body)
        self.assertRaises(webob.exc.HTTPBadRequest, data.stack_name)

    def test_template_inline(self):
        template = {'foo': 'bar', 'blarg': 'wibble'}
        body = {'template': template}
        data = stacks.InstantiationData(body)
        self.assertEqual(template, data.template())

    def test_template_string_json(self):
        template = '{"heat_template_version": "2013-05-23","foo": "bar", "blarg": "wibble"}'
        body = {'template': template}
        data = stacks.InstantiationData(body)
        self.assertEqual(json.loads(template), data.template())

    def test_template_string_yaml(self):
        template = 'HeatTemplateFormatVersion: 2012-12-12\nfoo: bar\nblarg: wibble\n'
        parsed = {u'HeatTemplateFormatVersion': u'2012-12-12', u'blarg': u'wibble', u'foo': u'bar'}
        body = {'template': template}
        data = stacks.InstantiationData(body)
        self.assertEqual(parsed, data.template())

    def test_template_int(self):
        template = '42'
        body = {'template': template}
        data = stacks.InstantiationData(body)
        self.assertRaises(webob.exc.HTTPBadRequest, data.template)

    def test_template_url(self):
        template = {'heat_template_version': '2013-05-23', 'foo': 'bar', 'blarg': 'wibble'}
        url = 'http://example.com/template'
        body = {'template_url': url}
        data = stacks.InstantiationData(body)
        mock_get = self.patchobject(urlfetch, 'get', return_value=json.dumps(template))
        self.assertEqual(template, data.template())
        mock_get.assert_called_once_with(url)

    def test_template_priority(self):
        template = {'foo': 'bar', 'blarg': 'wibble'}
        url = 'http://example.com/template'
        body = {'template': template, 'template_url': url}
        data = stacks.InstantiationData(body)
        mock_get = self.patchobject(urlfetch, 'get')
        self.assertEqual(template, data.template())
        mock_get.assert_not_called()

    def test_template_missing(self):
        template = {'foo': 'bar', 'blarg': 'wibble'}
        body = {'not the template': template}
        data = stacks.InstantiationData(body)
        self.assertRaises(webob.exc.HTTPBadRequest, data.template)

    def test_template_exceeds_max_template_size(self):
        cfg.CONF.set_override('max_template_size', 10)
        template = json.dumps(['a'] * cfg.CONF.max_template_size)
        body = {'template': template}
        data = stacks.InstantiationData(body)
        error = self.assertRaises(heat_exc.RequestLimitExceeded, data.template)
        msg = 'Request limit exceeded: Template size (%(actual_len)s bytes) exceeds maximum allowed size (%(limit)s bytes).' % {'actual_len': len(str(template)), 'limit': cfg.CONF.max_template_size}
        self.assertEqual(msg, str(error))

    def test_parameters(self):
        params = {'foo': 'bar', 'blarg': 'wibble'}
        body = {'parameters': params, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}
        data = stacks.InstantiationData(body)
        self.assertEqual(body, data.environment())

    def test_environment_only_params(self):
        env = {'parameters': {'foo': 'bar', 'blarg': 'wibble'}}
        body = {'environment': env}
        data = stacks.InstantiationData(body)
        self.assertEqual(env, data.environment())

    def test_environment_with_env_files(self):
        env = {'parameters': {'foo': 'bar', 'blarg': 'wibble'}}
        body = {'environment': env, 'environment_files': ['env.yaml']}
        expect = {'parameters': {}, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}
        data = stacks.InstantiationData(body)
        self.assertEqual(expect, data.environment())

    def test_environment_and_parameters(self):
        body = {'parameters': {'foo': 'bar'}, 'environment': {'parameters': {'blarg': 'wibble'}}}
        expect = {'parameters': {'blarg': 'wibble', 'foo': 'bar'}, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}
        data = stacks.InstantiationData(body)
        self.assertEqual(expect, data.environment())

    def test_parameters_override_environment(self):
        body = {'parameters': {'foo': 'bar', 'tester': 'Yes'}, 'environment': {'parameters': {'blarg': 'wibble', 'tester': 'fail'}}}
        expect = {'parameters': {'blarg': 'wibble', 'foo': 'bar', 'tester': 'Yes'}, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}
        data = stacks.InstantiationData(body)
        self.assertEqual(expect, data.environment())

    def test_environment_empty_params(self):
        env = {'parameters': None}
        body = {'environment': env}
        data = stacks.InstantiationData(body)
        self.assertRaises(webob.exc.HTTPBadRequest, data.environment)

    def test_environment_bad_format(self):
        env = {'somethingnotsupported': {'blarg': 'wibble'}}
        body = {'environment': json.dumps(env)}
        data = stacks.InstantiationData(body)
        self.assertRaises(webob.exc.HTTPBadRequest, data.environment)

    def test_environment_missing(self):
        env = {'foo': 'bar', 'blarg': 'wibble'}
        body = {'not the environment': env}
        data = stacks.InstantiationData(body)
        self.assertEqual({'parameters': {}, 'encrypted_param_names': [], 'parameter_defaults': {}, 'resource_registry': {}, 'event_sinks': []}, data.environment())

    def test_args(self):
        body = {'parameters': {}, 'environment': {}, 'stack_name': 'foo', 'template': {}, 'template_url': 'http://example.com/', 'timeout_mins': 60}
        data = stacks.InstantiationData(body)
        self.assertEqual({'timeout_mins': 60}, data.args())