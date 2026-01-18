import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class AwsStackTest(functional_base.FunctionalTestsBase):
    test_template = '\nHeatTemplateFormatVersion: \'2012-12-12\'\nResources:\n  the_nested:\n    Type: AWS::CloudFormation::Stack\n    Properties:\n      TemplateURL: the.yaml\n      Parameters:\n        KeyName: foo\nOutputs:\n  output_foo:\n    Value: {"Fn::GetAtt": [the_nested, Outputs.Foo]}\n'
    nested_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  KeyName:\n    Type: String\nOutputs:\n  Foo:\n    Value: bar\n"
    update_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  KeyName:\n    Type: String\nOutputs:\n  Foo:\n    Value: foo\n"
    nested_with_res_template = '\nHeatTemplateFormatVersion: \'2012-12-12\'\nParameters:\n  KeyName:\n    Type: String\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\nOutputs:\n  Foo:\n    Value: {"Fn::GetAtt": [NestedResource, value]}\n'

    def setUp(self):
        super(AwsStackTest, self).setUp()
        if not self.is_service_available('object-store'):
            self.skipTest('object-store service not available, skipping')
        self.object_container_name = test.rand_name()
        self.project_id = self.identity_client.project_id
        self.swift_key = hashlib.sha224(str(random.getrandbits(256)).encode('ascii')).hexdigest()[:32]
        key_header = 'x-container-meta-temp-url-key'
        self.object_client.put_container(self.object_container_name, {key_header: self.swift_key})
        self.addCleanup(self.object_client.delete_container, self.object_container_name)

    def publish_template(self, contents, cleanup=True):
        oc = self.object_client
        oc.put_object(self.object_container_name, 'template.yaml', contents)
        if cleanup:
            self.addCleanup(oc.delete_object, self.object_container_name, 'template.yaml')
        path = '/v1/AUTH_%s/%s/%s' % (self.project_id, self.object_container_name, 'template.yaml')
        timeout = self.conf.build_timeout * 10
        tempurl = swiftclient_utils.generate_temp_url(path, timeout, self.swift_key, 'GET')
        sw_url = parse.urlparse(oc.url)
        return '%s://%s%s' % (sw_url.scheme, sw_url.netloc, tempurl)

    def test_nested_stack_create(self):
        url = self.publish_template(self.nested_template)
        self.template = self.test_template.replace('the.yaml', url)
        stack_identifier = self.stack_create(template=self.template)
        stack = self.client.stacks.get(stack_identifier)
        self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        self.assertEqual('bar', self._stack_output(stack, 'output_foo'))

    def test_nested_stack_create_with_timeout(self):
        url = self.publish_template(self.nested_template)
        self.template = self.test_template.replace('the.yaml', url)
        timeout_template = yaml.safe_load(self.template)
        props = timeout_template['Resources']['the_nested']['Properties']
        props['TimeoutInMinutes'] = '50'
        stack_identifier = self.stack_create(template=timeout_template)
        stack = self.client.stacks.get(stack_identifier)
        self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        self.assertEqual('bar', self._stack_output(stack, 'output_foo'))

    def test_nested_stack_adopt_ok(self):
        url = self.publish_template(self.nested_with_res_template)
        self.template = self.test_template.replace('the.yaml', url)
        adopt_data = {'resources': {'the_nested': {'resource_id': 'test-res-id', 'resources': {'NestedResource': {'type': 'OS::Heat::RandomString', 'resource_id': 'test-nested-res-id', 'resource_data': {'value': 'goopie'}}}}}, 'environment': {'parameters': {}}, 'template': yaml.safe_load(self.template)}
        stack_identifier = self.stack_adopt(adopt_data=json.dumps(adopt_data))
        self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('goopie', self._stack_output(stack, 'output_foo'))

    def test_nested_stack_adopt_fail(self):
        url = self.publish_template(self.nested_with_res_template)
        self.template = self.test_template.replace('the.yaml', url)
        adopt_data = {'resources': {'the_nested': {'resource_id': 'test-res-id', 'resources': {}}}, 'environment': {'parameters': {}}, 'template': yaml.safe_load(self.template)}
        stack_identifier = self.stack_adopt(adopt_data=json.dumps(adopt_data), wait_for_status='ADOPT_FAILED')
        rsrc = self.client.resources.get(stack_identifier, 'the_nested')
        self.assertEqual('ADOPT_FAILED', rsrc.resource_status)

    def test_nested_stack_update(self):
        url = self.publish_template(self.nested_template)
        self.template = self.test_template.replace('the.yaml', url)
        stack_identifier = self.stack_create(template=self.template)
        original_nested_id = self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('bar', self._stack_output(stack, 'output_foo'))
        new_template = yaml.safe_load(self.template)
        props = new_template['Resources']['the_nested']['Properties']
        props['TemplateURL'] = self.publish_template(self.update_template, cleanup=False)
        self.update_stack(stack_identifier, new_template)
        new_nested_id = self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        self.assertEqual(original_nested_id, new_nested_id)
        updt_stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('foo', self._stack_output(updt_stack, 'output_foo'))

    def test_nested_stack_suspend_resume(self):
        url = self.publish_template(self.nested_template)
        self.template = self.test_template.replace('the.yaml', url)
        stack_identifier = self.stack_create(template=self.template)
        self.stack_suspend(stack_identifier)
        self.stack_resume(stack_identifier)