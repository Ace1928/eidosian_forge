from heat_integrationtests.functional import functional_base
class SimpleStackValidationTest(functional_base.FunctionalTestsBase):

    def test_validate_json_content(self):
        template = u'\nheat_template_version: rocky\nresources:\n  server:\n    type: OS::Heat::TestResource\n    properties:\n      value: =%da\n'
        self.stack_create(template=template, expected_status='CREATE_COMPLETE')