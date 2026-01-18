import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestUserPass(ShellBase):

    def setUp(self):
        super(ShellTestUserPass, self).setUp()
        if self.client is None:
            self.client = http.SessionClient
        self._set_fake_env()

    def _set_fake_env(self):
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_stack_list(self):
        self.register_keystone_auth_fixture()
        self.mock_stack_list()
        list_text = self.shell('stack-list')
        required = ['id', 'stack_status', 'creation_time', 'teststack', '1', 'CREATE_COMPLETE', 'IN_PROGRESS']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'parent')

    def test_stack_list_show_nested(self):
        self.register_keystone_auth_fixture()
        expected_url = '/stacks?%s' % parse.urlencode({'show_nested': True}, True)
        self.mock_stack_list(expected_url, show_nested=True)
        list_text = self.shell('stack-list --show-nested')
        required = ['teststack', 'teststack2', 'teststack_nested', 'parent', 'theparentof3']
        for r in required:
            self.assertRegex(list_text, r)

    def test_stack_list_show_owner(self):
        self.register_keystone_auth_fixture()
        self.mock_stack_list()
        list_text = self.shell('stack-list --show-owner')
        required = ['stack_owner', 'testowner']
        for r in required:
            self.assertRegex(list_text, r)

    def test_parsable_error(self):
        self.register_keystone_auth_fixture()
        message = 'The Stack (bad) could not be found.'
        self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(message))
        e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
        self.assertEqual('ERROR: ' + message, str(e))

    def test_parsable_verbose(self):
        self.register_keystone_auth_fixture()
        message = 'The Stack (bad) could not be found.'
        self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(message))
        exc.verbose = 1
        e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
        self.assertIn(message, str(e))

    def test_parsable_malformed_error(self):
        self.register_keystone_auth_fixture()
        invalid_json = 'ERROR: {Invalid JSON Error.'
        self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(invalid_json))
        e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
        self.assertEqual('ERROR: ' + invalid_json, str(e))

    def test_parsable_malformed_error_missing_message(self):
        self.register_keystone_auth_fixture()
        message = 'Internal Error'
        self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(message))
        e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
        self.assertEqual('ERROR: Internal Error', str(e))

    def test_parsable_malformed_error_missing_traceback(self):
        self.register_keystone_auth_fixture()
        message = 'The Stack (bad) could not be found.'
        self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(message))
        exc.verbose = 1
        e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
        self.assertEqual('ERROR: The Stack (bad) could not be found.\n', str(e))

    def test_stack_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z', 'tags': ['tag1', 'tag2']}}
        self.mock_request_get('/stacks/teststack/1', resp_dict)
        list_text = self.shell('stack-show teststack/1')
        required = ['id', 'stack_name', 'stack_status', 'creation_time', 'tags', 'teststack', 'CREATE_COMPLETE', '2012-10-25T01:58:47Z', "['tag1', 'tag2']"]
        for r in required:
            self.assertRegex(list_text, r)

    def test_stack_show_without_outputs(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}}
        params = {'resolve_outputs': False}
        self.mock_request_get('/stacks/teststack/1', resp_dict, params=params)
        list_text = self.shell('stack-show teststack/1 --no-resolve-outputs')
        required = ['id', 'stack_name', 'stack_status', 'creation_time', 'teststack', 'CREATE_COMPLETE', '2012-10-25T01:58:47Z']
        for r in required:
            self.assertRegex(list_text, r)

    def _output_fake_response(self, output_key):
        outputs = [{'output_value': 'value1', 'output_key': 'output1', 'description': 'test output 1'}, {'output_value': ['output', 'value', '2'], 'output_key': 'output2', 'description': 'test output 2'}, {'output_value': u'test♥', 'output_key': 'output_uni', 'description': 'test output unicode'}]

        def find_output(key):
            for out in outputs:
                if out['output_key'] == key:
                    return {'output': out}
        self.mock_request_get('/stacks/teststack/1/outputs/%s' % output_key, find_output(output_key))

    def _error_output_fake_response(self, output_key):
        resp_dict = {'output': {'output_value': 'null', 'output_key': 'output1', 'description': 'test output 1', 'output_error': 'The Referenced Attribute (0 PublicIP) is incorrect.'}}
        self.mock_request_get('/stacks/teststack/1/outputs/%s' % output_key, resp_dict)

    def test_template_show_cfn(self):
        self.register_keystone_auth_fixture()
        template_data = open(os.path.join(TEST_VAR_DIR, 'minimal.template')).read()
        resp_dict = jsonutils.loads(template_data)
        self.mock_request_get('/stacks/teststack/template', resp_dict)
        show_text = self.shell('template-show teststack')
        required = ['{', '  "AWSTemplateFormatVersion": "2010-09-09"', '  "Outputs": {}', '  "Resources": {}', '  "Parameters": {}', '}']
        for r in required:
            self.assertRegex(show_text, r)

    def test_template_show_cfn_unicode(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'AWSTemplateFormatVersion': '2010-09-09', 'Description': u'test♥', 'Outputs': {}, 'Resources': {}, 'Parameters': {}}
        self.mock_request_get('/stacks/teststack/template', resp_dict)
        show_text = self.shell('template-show teststack')
        required = ['{', '  "AWSTemplateFormatVersion": "2010-09-09"', '  "Outputs": {}', '  "Parameters": {}', '  "Description": "test♥"', '  "Resources": {}', '}']
        for r in required:
            self.assertRegex(show_text, r)

    def test_template_show_hot(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'heat_template_version': '2013-05-23', 'parameters': {}, 'resources': {}, 'outputs': {}}
        self.mock_request_get('/stacks/teststack/template', resp_dict)
        show_text = self.shell('template-show teststack')
        required = ["heat_template_version: '2013-05-23'", 'outputs: {}', 'parameters: {}', 'resources: {}']
        for r in required:
            self.assertRegex(show_text, r)

    def test_template_validate(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'heat_template_version': '2013-05-23', 'parameters': {}, 'resources': {}, 'outputs': {}}
        self.mock_request_post('/validate', resp_dict, data=mock.ANY)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        cmd = 'template-validate -f %s -P foo=bar' % template_file
        show_text = self.shell(cmd)
        required = ['heat_template_version', 'outputs', 'parameters', 'resources']
        for r in required:
            self.assertRegex(show_text, r)

    def _test_stack_preview(self, timeout=None, enable_rollback=False, tags=None):
        self.register_keystone_auth_fixture()
        resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'resources': {'1': {'name': 'r1'}}, 'creation_time': '2012-10-25T01:58:47Z', 'timeout_mins': timeout, 'disable_rollback': not enable_rollback, 'tags': tags}}
        self.mock_request_post('/stacks/preview', resp_dict, data=mock.ANY, req_headers=True)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        cmd = 'stack-preview teststack --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17" ' % template_file
        if enable_rollback:
            cmd += '-r '
        if timeout:
            cmd += '--timeout=%d ' % timeout
        if tags:
            cmd += '--tags=%s ' % tags
        preview_text = self.shell(cmd)
        required = ['stack_name', 'id', 'teststack', '1', 'resources', 'timeout_mins', 'disable_rollback', 'tags']
        for r in required:
            self.assertRegex(preview_text, r)

    def test_stack_preview(self):
        self._test_stack_preview()

    def test_stack_preview_timeout(self):
        self._test_stack_preview(300, True)

    def test_stack_preview_tags(self):
        self._test_stack_preview(tags='tag1,tag2')

    def test_stack_create(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        create_text = self.shell('stack-create teststack --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(create_text, r)

    def test_create_success_with_poll(self):
        self.register_keystone_auth_fixture()
        stack_create_resp_dict = {'stack': {'id': 'teststack2/2', 'stack_name': 'teststack2', 'stack_status': 'CREATE_IN_PROGRESS', 'creation_time': '2012-10-25T01:58:47Z'}}
        self.mock_request_post('/stacks', stack_create_resp_dict, data=mock.ANY, req_headers=True, status_code=201)
        self.mock_stack_list()
        stack_show_resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}}
        event_list_resp_dict = self.event_list_resp_dict(stack_name='teststack2')
        stack_id = 'teststack2'
        self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, event_list_resp_dict)
        self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        create_text = self.shell('stack-create teststack2 --poll 4 --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['id', 'stack_name', 'stack_status', '2', 'teststack2', 'IN_PROGRESS', '14:14:30', '2013-12-05', 'CREATE_IN_PROGRESS', 'state changed', '14:14:31', 'testresource', '14:14:32', 'CREATE_COMPLETE', '14:14:33']
        for r in required:
            self.assertRegex(create_text, r)

    def test_create_failed_with_poll(self):
        self.register_keystone_auth_fixture()
        stack_create_resp_dict = {'stack': {'id': 'teststack2/2', 'stack_name': 'teststack2', 'stack_status': 'CREATE_IN_PROGRESS', 'creation_time': '2012-10-25T01:58:47Z'}}
        self.mock_request_post('/stacks', stack_create_resp_dict, data=mock.ANY, req_headers=True, status_code=201)
        self.mock_stack_list()
        stack_show_resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}}
        event_list_resp_dict = self.event_list_resp_dict(stack_name='teststack2', final_state='FAILED')
        stack_id = 'teststack2'
        self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, event_list_resp_dict)
        self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        e = self.assertRaises(exc.StackFailure, self.shell, 'stack-create teststack2 --poll --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=password;KeyName=heat_key;LinuxDistribution=F17' % template_file)
        self.assertEqual('\n Stack teststack2 CREATE_FAILED \n', str(e))

    def test_stack_create_param_file(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        self.useFixture(fixtures.MockPatchObject(utils, 'read_url_content', return_value='xxxxxx'))
        url = 'file://' + request.pathname2url('%s/private_key.env' % TEST_VAR_DIR)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        create_text = self.shell('stack-create teststack --template-file=%s --parameter-file private_key=private_key.env --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(create_text, r)
        utils.read_url_content.assert_called_once_with(url)

    def test_stack_create_only_param_file(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        self.useFixture(fixtures.MockPatchObject(utils, 'read_url_content', return_value='xxxxxx'))
        url = 'file://' + request.pathname2url('%s/private_key.env' % TEST_VAR_DIR)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        create_text = self.shell('stack-create teststack --template-file=%s --parameter-file private_key=private_key.env ' % template_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(create_text, r)
        utils.read_url_content.assert_called_once_with(url)

    def test_stack_create_timeout(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'disable_rollback': True, 'parameters': {'DBUsername': 'wp', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17"', '"InstanceType': 'm1.large', 'DBPassword': 'verybadpassword'}, 'stack_name': 'teststack', 'environment': {}, 'template': jsonutils.loads(template_data), 'timeout_mins': 123}
        self.mock_request_post('/stacks', None, data=expected_data, status_code=201, req_headers=True)
        self.mock_stack_list()
        create_text = self.shell('stack-create teststack --template-file=%s --timeout=123 --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(create_text, r)

    def test_stack_update_timeout(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'DBUsername': 'wp', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17"', '"InstanceType': 'm1.large', 'DBPassword': 'verybadpassword'}, 'timeout_mins': 123, 'disable_rollback': True}
        self.mock_request_put('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --timeout 123 --rollback off --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_create_url(self):
        self.register_keystone_auth_fixture()
        url_content = io.StringIO('{"AWSTemplateFormatVersion" : "2010-09-09"}')
        self.useFixture(fixtures.MockPatchObject(request, 'urlopen', return_value=url_content))
        expected_data = {'files': {}, 'disable_rollback': True, 'stack_name': 'teststack', 'environment': {}, 'template': {'AWSTemplateFormatVersion': '2010-09-09'}, 'parameters': {'DBUsername': 'wp', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17"', '"InstanceType': 'm1.large', 'DBPassword': 'verybadpassword'}}
        self.mock_request_post('/stacks', None, data=expected_data, status_code=201, req_headers=True)
        self.mock_stack_list()
        create_text = self.shell('stack-create teststack --template-url=http://no.where/minimal.template --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"')
        required = ['stack_name', 'id', 'teststack2', '2']
        for r in required:
            self.assertRegex(create_text, r)
        request.urlopen.assert_called_once_with('http://no.where/minimal.template')

    def test_stack_create_object(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        self.mock_request_get('http://no.where/container/minimal.template', template_data, raw=True)
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        create_text = self.shell('stack-create teststack2 --template-object=http://no.where/container/minimal.template --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"')
        required = ['stack_name', 'id', 'teststack2', '2']
        for r in required:
            self.assertRegex(create_text, r)

    def test_stack_create_with_tags(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'disable_rollback': True, 'parameters': {'DBUsername': 'wp', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17"', '"InstanceType': 'm1.large', 'DBPassword': 'verybadpassword'}, 'stack_name': 'teststack', 'environment': {}, 'template': jsonutils.loads(template_data), 'tags': 'tag1,tag2'}
        self.mock_request_post('/stacks', None, data=expected_data, status_code=201, req_headers=True)
        self.mock_stack_list()
        create_text = self.shell('stack-create teststack --template-file=%s --tags=tag1,tag2 --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(create_text, r)

    def test_stack_abandon(self):
        self.register_keystone_auth_fixture()
        abandoned_stack = {'action': 'CREATE', 'status': 'COMPLETE', 'name': 'teststack', 'id': '1', 'resources': {'foo': {'name': 'foo', 'resource_id': 'test-res-id', 'action': 'CREATE', 'status': 'COMPLETE', 'resource_data': {}, 'metadata': {}}}}
        self.mock_request_delete('/stacks/teststack/1/abandon', abandoned_stack)
        abandon_resp = self.shell('stack-abandon teststack/1')
        self.assertEqual(abandoned_stack, jsonutils.loads(abandon_resp))

    def test_stack_abandon_with_outputfile(self):
        self.register_keystone_auth_fixture()
        abandoned_stack = {'action': 'CREATE', 'status': 'COMPLETE', 'name': 'teststack', 'id': '1', 'resources': {'foo': {'name': 'foo', 'resource_id': 'test-res-id', 'action': 'CREATE', 'status': 'COMPLETE', 'resource_data': {}, 'metadata': {}}}}
        self.mock_request_delete('/stacks/teststack/1/abandon', abandoned_stack)
        with tempfile.NamedTemporaryFile() as file_obj:
            self.shell('stack-abandon teststack/1 -O %s' % file_obj.name)
            result = jsonutils.loads(file_obj.read().decode())
            self.assertEqual(abandoned_stack, result)

    def test_stack_adopt(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        adopt_data_file = os.path.join(TEST_VAR_DIR, 'adopt_stack_data.json')
        adopt_text = self.shell('stack-adopt teststack --adopt-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % adopt_data_file)
        required = ['stack_name', 'id', 'teststack', '1']
        for r in required:
            self.assertRegex(adopt_text, r)

    def test_stack_adopt_with_environment(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
        self.mock_stack_list()
        adopt_data_file = os.path.join(TEST_VAR_DIR, 'adopt_stack_data.json')
        environment_file = os.path.join(TEST_VAR_DIR, 'environment.json')
        self.shell('stack-adopt teststack --adopt-file=%s --environment-file=%s' % (adopt_data_file, environment_file))

    def test_stack_adopt_without_data(self):
        self.register_keystone_auth_fixture()
        failed_msg = 'Need to specify --adopt-file'
        self.shell_error('stack-adopt teststack ', failed_msg, exception=exc.CommandError)

    def test_stack_adopt_empty_data_file(self):
        failed_msg = 'Invalid adopt-file, no data!'
        self.register_keystone_auth_fixture()
        with tempfile.NamedTemporaryFile() as file_obj:
            self.shell_error('stack-adopt teststack --adopt-file=%s ' % file_obj.name, failed_msg, exception=exc.CommandError)

    def test_stack_update_enable_rollback(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        with open(template_file, 'rb') as f:
            template_data = jsonutils.load(f)
        expected_data = {'files': {}, 'environment': {}, 'template': template_data, 'disable_rollback': False, 'parameters': mock.ANY}
        self.mock_request_put('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --rollback on --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_disable_rollback(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        with open(template_file, 'rb') as f:
            template_data = jsonutils.load(f)
        expected_data = {'files': {}, 'environment': {}, 'template': template_data, 'disable_rollback': True, 'parameters': mock.ANY}
        self.mock_request_put('/stacks/teststack2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2 --template-file=%s --rollback off --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_fault_rollback_value(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        self.shell_error('stack-update teststack2/2 --rollback Foo --template-file=%s' % template_file, "Unrecognized value 'Foo', acceptable values are:", exception=exc.CommandError)

    def test_stack_update_rollback_default(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        with open(template_file, 'rb') as f:
            template_data = jsonutils.load(f)
        expected_data = {'files': {}, 'environment': {}, 'template': template_data, 'parameters': mock.ANY}
        self.mock_request_put('/stacks/teststack2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2 --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
        required = ['stack_name', 'id', 'teststack2', '2']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_existing_parameters(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {}, 'disable_rollback': False}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_patched_existing_parameters(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'disable_rollback': False}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --parameters="KeyPairName=updated_key" --existing' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_existing_and_default_parameters(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {}, 'clear_parameters': ['InstanceType', 'DBUsername', 'DBPassword', 'KeyPairName', 'LinuxDistribution'], 'disable_rollback': False}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing --clear-parameter=InstanceType --clear-parameter=DBUsername --clear-parameter=DBPassword --clear-parameter=KeyPairName --clear-parameter=LinuxDistribution' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_patched_and_default_parameters(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'clear_parameters': ['InstanceType', 'DBUsername', 'DBPassword', 'KeyPairName', 'LinuxDistribution'], 'disable_rollback': False}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing --parameters="KeyPairName=updated_key" --clear-parameter=InstanceType --clear-parameter=DBUsername --clear-parameter=DBPassword --clear-parameter=KeyPairName --clear-parameter=LinuxDistribution' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_existing_template(self):
        self.register_keystone_auth_fixture()
        expected_data = {'files': {}, 'environment': {}, 'template': None, 'parameters': {}}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --existing')
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_update_with_tags(self):
        self.register_keystone_auth_fixture()
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        template_data = open(template_file).read()
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'disable_rollback': False, 'tags': 'tag1,tag2'}
        self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
        self.mock_stack_list()
        update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing --parameters="KeyPairName=updated_key" --tags=tag1,tag2 ' % template_file)
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def _setup_stubs_update_dry_run(self, template_file, existing=False, show_nested=False):
        self.register_keystone_auth_fixture()
        template_data = open(template_file).read()
        replaced_res = {'resource_name': 'my_res', 'resource_identity': {'stack_name': 'teststack2', 'stack_id': '2', 'tenant': '1234', 'path': '/resources/my_res'}, 'description': '', 'stack_identity': {'stack_name': 'teststack2', 'stack_id': '2', 'tenant': '1234', 'path': ''}, 'stack_name': 'teststack2', 'creation_time': '2015-08-19T19:43:34.025507', 'resource_status': 'COMPLETE', 'updated_time': '2015-08-19T19:43:34.025507', 'resource_type': 'OS::Heat::RandomString', 'required_by': [], 'resource_status_reason': '', 'physical_resource_id': '', 'attributes': {'value': None}, 'resource_action': 'INIT', 'metadata': {}}
        resp_dict = {'resource_changes': {'deleted': [], 'unchanged': [], 'added': [], 'replaced': [replaced_res], 'updated': []}}
        expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'disable_rollback': False}
        if show_nested:
            path = '/stacks/teststack2/2/preview?show_nested=True'
        else:
            path = '/stacks/teststack2/2/preview'
        if existing:
            self.mock_request_patch(path, resp_dict, data=expected_data)
        else:
            self.mock_request_put(path, resp_dict, data=expected_data)

    def test_stack_update_dry_run(self):
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        self._setup_stubs_update_dry_run(template_file)
        update_preview_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --parameters="KeyPairName=updated_key" --dry-run ' % template_file)
        required = ['stack_name', 'id', 'teststack2', '2', 'state', 'replaced']
        for r in required:
            self.assertRegex(update_preview_text, r)

    def test_stack_update_dry_run_show_nested(self):
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        self._setup_stubs_update_dry_run(template_file, show_nested=True)
        update_preview_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --show-nested --parameters="KeyPairName=updated_key" --dry-run ' % template_file)
        required = ['stack_name', 'id', 'teststack2', '2', 'state', 'replaced']
        for r in required:
            self.assertRegex(update_preview_text, r)

    def test_stack_update_dry_run_patch(self):
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        self._setup_stubs_update_dry_run(template_file, existing=True)
        update_preview_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing --parameters="KeyPairName=updated_key" --dry-run ' % template_file)
        required = ['stack_name', 'id', 'teststack2', '2', 'state', 'replaced']
        for r in required:
            self.assertRegex(update_preview_text, r)

    @mock.patch('sys.stdin', new_callable=io.StringIO)
    def test_stack_delete_prompt_with_tty(self, ms):
        self.register_keystone_auth_fixture()
        mock_stdin = mock.Mock()
        mock_stdin.isatty = mock.Mock()
        mock_stdin.isatty.return_value = True
        mock_stdin.readline = mock.Mock()
        mock_stdin.readline.return_value = 'n'
        mock_stdin.fileno.return_value = 0
        sys.stdin = mock_stdin
        self.mock_request_delete('/stacks/teststack2/2', None)
        resp = self.shell('stack-delete teststack2/2')
        resp_text = 'Are you sure you want to delete this stack(s) [y/N]? '
        self.assertEqual(resp_text, resp)
        mock_stdin.readline.return_value = 'y'
        resp = self.shell('stack-delete teststack2/2')
        msg = 'Request to delete stack teststack2/2 has been accepted.'
        self.assertRegex(resp, msg)

    @mock.patch('sys.stdin', new_callable=io.StringIO)
    def test_stack_delete_prompt_with_tty_y(self, ms):
        self.register_keystone_auth_fixture()
        mock_stdin = mock.Mock()
        mock_stdin.isatty = mock.Mock()
        mock_stdin.isatty.return_value = True
        mock_stdin.readline = mock.Mock()
        mock_stdin.readline.return_value = ''
        mock_stdin.fileno.return_value = 0
        sys.stdin = mock_stdin
        self.mock_request_delete('/stacks/teststack2/2')
        resp = self.shell('stack-delete -y teststack2/2')
        msg = 'Request to delete stack teststack2/2 has been accepted.'
        self.assertRegex(resp, msg)

    def test_stack_delete(self):
        self.register_keystone_auth_fixture()
        self.mock_request_delete('/stacks/teststack2/2')
        resp = self.shell('stack-delete teststack2/2')
        msg = 'Request to delete stack teststack2/2 has been accepted.'
        self.assertRegex(resp, msg)

    def test_stack_delete_multiple(self):
        self.register_keystone_auth_fixture()
        self.mock_request_delete('/stacks/teststack/1')
        self.mock_request_delete('/stacks/teststack2/2')
        resp = self.shell('stack-delete teststack/1 teststack2/2')
        msg1 = 'Request to delete stack teststack/1 has been accepted.'
        msg2 = 'Request to delete stack teststack2/2 has been accepted.'
        self.assertRegex(resp, msg1)
        self.assertRegex(resp, msg2)

    def test_stack_delete_failed_on_notfound(self):
        self.register_keystone_auth_fixture()
        self.mock_request_error('/stacks/teststack1/1', 'DELETE', exc.HTTPNotFound())
        error = self.assertRaises(exc.CommandError, self.shell, 'stack-delete teststack1/1')
        self.assertIn('Unable to delete 1 of the 1 stacks.', str(error))

    def test_stack_delete_failed_on_forbidden(self):
        self.register_keystone_auth_fixture()
        self.mock_request_error('/stacks/teststack1/1', 'DELETE', exc.Forbidden())
        error = self.assertRaises(exc.CommandError, self.shell, 'stack-delete teststack1/1')
        self.assertIn('Unable to delete 1 of the 1 stacks.', str(error))

    def test_build_info(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'build_info': {'api': {'revision': 'api_revision'}, 'engine': {'revision': 'engine_revision'}}}
        self.mock_request_get('/build_info', resp_dict)
        build_info_text = self.shell('build-info')
        required = ['api', 'engine', 'revision', 'api_revision', 'engine_revision']
        for r in required:
            self.assertRegex(build_info_text, r)

    def test_stack_snapshot(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshot': {'id': '1', 'creation_time': '2012-10-25T01:58:47Z'}}
        self.mock_request_post('/stacks/teststack/1/snapshots', resp_dict, data={})
        resp = self.shell('stack-snapshot teststack/1')
        self.assertEqual(resp_dict, jsonutils.loads(resp))

    def test_snapshot_list(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshots': [{'id': '2', 'name': 'snap1', 'status': 'COMPLETE', 'status_reason': '', 'creation_time': '2014-12-05T01:25:52Z'}]}
        self.mock_request_get('/stacks/teststack/1/snapshots', resp_dict)
        list_text = self.shell('snapshot-list teststack/1')
        required = ['id', 'name', 'status', 'status_reason', 'creation_time', '2', 'COMPLETE', '2014-12-05T01:25:52Z']
        for r in required:
            self.assertRegex(list_text, r)

    def test_snapshot_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshot': {'id': '2', 'creation_time': '2012-10-25T01:58:47Z'}}
        self.mock_request_get('/stacks/teststack/1/snapshots/2', resp_dict)
        resp = self.shell('snapshot-show teststack/1 2')
        self.assertEqual(resp_dict, jsonutils.loads(resp))

    @mock.patch('sys.stdin', new_callable=io.StringIO)
    def test_snapshot_delete_prompt_with_tty(self, ms):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshot': {'id': '2', 'creation_time': '2012-10-25T01:58:47Z'}}
        mock_stdin = mock.Mock()
        mock_stdin.isatty = mock.Mock()
        mock_stdin.isatty.return_value = True
        mock_stdin.readline = mock.Mock()
        mock_stdin.readline.return_value = 'n'
        sys.stdin = mock_stdin
        self.mock_request_delete('/stacks/teststack/1/snapshots/2', resp_dict)
        resp = self.shell('snapshot-delete teststack/1 2')
        resp_text = 'Are you sure you want to delete the snapshot of this stack [Y/N]?'
        self.assertEqual(resp_text, resp)
        mock_stdin.readline.return_value = 'Y'
        resp = self.shell('snapshot-delete teststack/1 2')
        msg = _('Request to delete the snapshot 2 of the stack teststack/1 has been accepted.')
        self.assertRegex(resp, msg)

    @mock.patch('sys.stdin', new_callable=io.StringIO)
    def test_snapshot_delete_prompt_with_tty_y(self, ms):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshot': {'id': '2', 'creation_time': '2012-10-25T01:58:47Z'}}
        mock_stdin = mock.Mock()
        mock_stdin.isatty = mock.Mock()
        mock_stdin.isatty.return_value = True
        mock_stdin.readline = mock.Mock()
        mock_stdin.readline.return_value = ''
        sys.stdin = mock_stdin
        self.mock_request_delete('/stacks/teststack/1/snapshots/2', resp_dict)
        resp = self.shell('snapshot-delete -y teststack/1 2')
        msg = _('Request to delete the snapshot 2 of the stack teststack/1 has been accepted.')
        self.assertRegex(resp, msg)

    def test_snapshot_delete(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'snapshot': {'id': '2', 'creation_time': '2012-10-25T01:58:47Z'}}
        self.mock_request_delete('/stacks/teststack/1/snapshots/2', resp_dict)
        resp = self.shell('snapshot-delete teststack/1 2')
        msg = _('Request to delete the snapshot 2 of the stack teststack/1 has been accepted.')
        self.assertRegex(resp, msg)

    def test_stack_restore(self):
        self.register_keystone_auth_fixture()
        self.mock_request_post('/stacks/teststack/1/snapshots/2/restore', None, status_code=204)
        resp = self.shell('stack-restore teststack/1 2')
        self.assertEqual('', resp)

    def test_output_list(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'outputs': [{'output_key': 'key', 'description': 'description'}, {'output_key': 'key1', 'description': 'description1'}]}
        self.mock_request_get('/stacks/teststack/1/outputs', resp_dict)
        list_text = self.shell('output-list teststack/1')
        required = ['output_key', 'description', 'key', 'description', 'key1', 'description1']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_list_api_400_error(self):
        self.register_keystone_auth_fixture()
        outputs = [{'output_key': 'key', 'description': 'description'}, {'output_key': 'key1', 'description': 'description1'}]
        stack_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z', 'outputs': outputs}}
        self.mock_request_error('/stacks/teststack/1/outputs', 'GET', exc.HTTPNotFound())
        self.mock_request_get('/stacks/teststack/1', stack_dict)
        list_text = self.shell('output-list teststack/1')
        required = ['output_key', 'description', 'key', 'description', 'key1', 'description1']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show_all(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'outputs': [{'output_key': 'key', 'description': 'description'}]}
        resp_dict1 = {'output': {'output_key': 'key', 'output_value': 'value', 'description': 'description'}}
        self.mock_request_get('/stacks/teststack/1/outputs', resp_dict)
        self.mock_request_get('/stacks/teststack/1/outputs/key', resp_dict1)
        list_text = self.shell('output-show --with-detail teststack/1 --all')
        required = ['output_key', 'output_value', 'description', 'key', 'value', 'description']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'output': {'output_key': 'key', 'output_value': 'value', 'description': 'description'}}
        self.mock_request_get('/stacks/teststack/1/outputs/key', resp_dict)
        resp = self.shell('output-show --with-detail teststack/1 key')
        required = ['output_key', 'output_value', 'description', 'key', 'value', 'description']
        for r in required:
            self.assertRegex(resp, r)

    def test_output_show_api_400_error(self):
        self.register_keystone_auth_fixture()
        output = {'output_key': 'key', 'output_value': 'value', 'description': 'description'}
        stack_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z', 'outputs': [output]}}
        self.mock_request_error('/stacks/teststack/1/outputs/key', 'GET', exc.HTTPNotFound())
        self.mock_request_get('/stacks/teststack/1', stack_dict)
        resp = self.shell('output-show --with-detail teststack/1 key')
        required = ['output_key', 'output_value', 'description', 'key', 'value', 'description']
        for r in required:
            self.assertRegex(resp, r)

    def test_output_show_output1_with_detail(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output1')
        list_text = self.shell('output-show teststack/1 output1 --with-detail')
        required = ['output_key', 'output_value', 'description', 'output1', 'value1', 'test output 1']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show_output1(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output1')
        list_text = self.shell('output-show -F raw teststack/1 output1')
        self.assertEqual('value1\n', list_text)

    def test_output_show_output2_raw(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output2')
        list_text = self.shell('output-show -F raw teststack/1 output2')
        self.assertEqual('[\n  "output", \n  "value", \n  "2"\n]\n', list_text)

    def test_output_show_output2_raw_with_detail(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output2')
        list_text = self.shell('output-show -F raw --with-detail teststack/1 output2')
        required = ['output_key', 'output_value', 'description', 'output2', "['output', 'value', '2']", 'test output 2']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show_output2_json(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output2')
        list_text = self.shell('output-show -F json teststack/1 output2')
        required = ['{', '"output_key": "output2"', '"description": "test output 2"', '"output_value": \\[', '"output"', '"value"', '"2"', '}']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show_output2_json_with_detail(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output2')
        list_text = self.shell('output-show -F json --with-detail teststack/1 output2')
        required = ['output_key', 'output_value', 'description', 'output2', '[\n    "output", \n    "value", \n    "2"\n  ]test output 2']
        for r in required:
            self.assertRegex(list_text, r)

    def test_output_show_unicode_output(self):
        self.register_keystone_auth_fixture()
        self._output_fake_response('output_uni')
        list_text = self.shell('output-show teststack/1 output_uni')
        self.assertEqual('test♥\n', list_text)

    def test_output_show_error(self):
        self.register_keystone_auth_fixture()
        self._error_output_fake_response('output1')
        error = self.assertRaises(exc.CommandError, self.shell, 'output-show teststack/1 output1')
        self.assertIn('The Referenced Attribute (0 PublicIP) is incorrect.', str(error))