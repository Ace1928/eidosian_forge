import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceCheckTest(functional_base.FunctionalTestsBase):
    """Prove that we can do template resource check."""
    main_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: my_name\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    nested_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: foo\n    Type: String\nResources:\n  RealRandom:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: one}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [RealRandom, value]}\n"

    def test_check(self):
        stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.nested_templ})
        self.client.actions.check(stack_id=stack_identifier)
        self._wait_for_stack_status(stack_identifier, 'CHECK_COMPLETE')