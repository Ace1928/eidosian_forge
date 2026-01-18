import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceSuspendResumeTest(functional_base.FunctionalTestsBase):
    """Prove that we can do template resource suspend/resume."""
    main_template = '\nheat_template_version: 2014-10-16\nparameters:\nresources:\n  the_nested:\n    type: the.yaml\n'
    nested_templ = '\nheat_template_version: 2014-10-16\nresources:\n  test_random_string:\n    type: OS::Heat::RandomString\n'

    def test_suspend_resume(self):
        """Basic test for template resource suspend resume."""
        stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.nested_templ})
        self.stack_suspend(stack_identifier=stack_identifier)
        self.stack_resume(stack_identifier=stack_identifier)