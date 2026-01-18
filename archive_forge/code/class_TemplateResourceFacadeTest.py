import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceFacadeTest(functional_base.FunctionalTestsBase):
    """Prove that we can use ResourceFacade in a HOT template."""
    main_template = '\nheat_template_version: 2013-05-23\nresources:\n  the_nested:\n    type: the.yaml\n    metadata:\n      foo: bar\noutputs:\n  value:\n    value: {get_attr: [the_nested, output]}\n'
    nested_templ = '\nheat_template_version: 2013-05-23\nresources:\n  test:\n    type: OS::Heat::TestResource\n    properties:\n      value: {"Fn::Select": [foo, {resource_facade: metadata}]}\noutputs:\n  output:\n    value: {get_attr: [test, output]}\n    '

    def test_metadata(self):
        stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.nested_templ})
        stack = self.client.stacks.get(stack_identifier)
        value = self._stack_output(stack, 'value')
        self.assertEqual('bar', value)