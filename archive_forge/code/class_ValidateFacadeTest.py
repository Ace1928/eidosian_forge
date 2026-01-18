import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class ValidateFacadeTest(functional_base.FunctionalTestsBase):
    """Prove that nested stack errors don't suck."""
    template = '\nheat_template_version: 2015-10-15\nresources:\n  thisone:\n    type: OS::Thingy\n    properties:\n      one: pre\n      two: post\noutputs:\n  one:\n    value: {get_attr: [thisone, here-it-is]}\n'
    templ_facade = '\nheat_template_version: 2015-04-30\nparameters:\n  one:\n    type: string\n  two:\n    type: string\noutputs:\n  here-it-is:\n    value: noop\n'
    env = '\nresource_registry:\n  OS::Thingy: facade.yaml\n  resources:\n    thisone:\n      OS::Thingy: concrete.yaml\n'

    def setUp(self):
        super(ValidateFacadeTest, self).setUp()
        self.client = self.orchestration_client

    def test_missing_param(self):
        templ_missing_parameter = '\nheat_template_version: 2015-04-30\nparameters:\n  one:\n    type: string\nresources:\n  str:\n    type: OS::Heat::RandomString\noutputs:\n  here-it-is:\n    value:\n      not-important\n'
        template = yaml.safe_load(self.template)
        del template['resources']['thisone']['properties']['two']
        try:
            self.stack_create(template=yaml.safe_dump(template), environment=self.env, files={'facade.yaml': self.templ_facade, 'concrete.yaml': templ_missing_parameter}, expected_status='CREATE_FAILED')
        except heat_exceptions.HTTPBadRequest as exc:
            exp = 'ERROR: Required property two for facade OS::Thingy missing in provider'
            self.assertEqual(exp, str(exc))

    def test_missing_output(self):
        templ_missing_output = '\nheat_template_version: 2015-04-30\nparameters:\n  one:\n    type: string\n  two:\n    type: string\nresources:\n  str:\n    type: OS::Heat::RandomString\n'
        try:
            self.stack_create(template=self.template, environment=self.env, files={'facade.yaml': self.templ_facade, 'concrete.yaml': templ_missing_output}, expected_status='CREATE_FAILED')
        except heat_exceptions.HTTPBadRequest as exc:
            exp = 'ERROR: Attribute here-it-is for facade OS::Thingy missing in provider'
            self.assertEqual(exp, str(exc))