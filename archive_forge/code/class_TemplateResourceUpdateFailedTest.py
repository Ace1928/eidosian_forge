import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceUpdateFailedTest(functional_base.FunctionalTestsBase):
    """Prove that we can do updates on a nested stack to fix a stack."""
    main_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  test:\n    Type: OS::Heat::TestResource\n    Properties:\n      fail: replace-this\n  server:\n    Type: server_fail.yaml\n    DependsOn: test\n"
    nested_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  RealRandom:\n    Type: OS::Heat::RandomString\n"

    def setUp(self):
        super(TemplateResourceUpdateFailedTest, self).setUp()

    def test_update_on_failed_create(self):
        broken_templ = self.main_template.replace('replace-this', 'true')
        stack_identifier = self.stack_create(template=broken_templ, files={'server_fail.yaml': self.nested_templ}, expected_status='CREATE_FAILED')
        fixed_templ = self.main_template.replace('replace-this', 'false')
        self.update_stack(stack_identifier, fixed_templ, files={'server_fail.yaml': self.nested_templ})