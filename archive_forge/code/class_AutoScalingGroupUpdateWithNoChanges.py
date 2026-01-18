from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class AutoScalingGroupUpdateWithNoChanges(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: 2013-05-23\n\nresources:\n  test_group:\n    type: OS::Heat::AutoScalingGroup\n    properties:\n      desired_capacity: 0\n      max_size: 0\n      min_size: 0\n      resource:\n        type: OS::Heat::RandomString\n  test_policy:\n    type: OS::Heat::ScalingPolicy\n    properties:\n      adjustment_type: change_in_capacity\n      auto_scaling_group_id: { get_resource: test_group }\n      scaling_adjustment: 1\n'

    def test_as_group_update_without_resource_changes(self):
        stack_identifier = self.stack_create(template=self.template)
        new_template = self.template.replace('scaling_adjustment: 1', 'scaling_adjustment: 2')
        self.update_stack(stack_identifier, template=new_template)