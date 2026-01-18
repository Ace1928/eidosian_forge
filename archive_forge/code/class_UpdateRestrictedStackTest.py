import copy
from heat_integrationtests.functional import functional_base
class UpdateRestrictedStackTest(functional_base.FunctionalTestsBase):

    def _check_for_restriction_reason(self, events, reason, num_expected=1):
        matched = [e for e in events if e.resource_status_reason == reason]
        return len(matched) == num_expected

    def test_update(self):
        stack_identifier = self.stack_create(template=test_template)
        update_template = copy.deepcopy(test_template)
        props = update_template['resources']['bar']['properties']
        props['value'] = '4567'
        self.update_stack(stack_identifier, update_template, env_both_restrict, expected_status='UPDATE_FAILED')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'CREATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_update_restrict))
        self.update_stack(stack_identifier, update_template, env_replace_restrict, expected_status='UPDATE_COMPLETE')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'UPDATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertFalse(self._check_for_restriction_reason(resource_events, reason_update_restrict, 2))
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_replace_restrict, 0))

    def test_replace(self):
        stack_identifier = self.stack_create(template=test_template)
        update_template = copy.deepcopy(test_template)
        props = update_template['resources']['bar']['properties']
        props['value'] = '4567'
        props['update_replace'] = True
        self.update_stack(stack_identifier, update_template, env_both_restrict, expected_status='UPDATE_FAILED')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'CREATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_replace_restrict))
        self.update_stack(stack_identifier, update_template, env_replace_restrict, expected_status='UPDATE_FAILED')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'CREATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_replace_restrict, 2))
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_update_restrict, 0))

    def test_update_type_changed(self):
        stack_identifier = self.stack_create(template=test_template)
        update_template = copy.deepcopy(test_template)
        rsrc = update_template['resources']['bar']
        rsrc['type'] = 'OS::Heat::None'
        self.update_stack(stack_identifier, update_template, env_both_restrict, expected_status='UPDATE_FAILED')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'CREATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_replace_restrict))
        self.update_stack(stack_identifier, update_template, env_replace_restrict, expected_status='UPDATE_FAILED')
        self.assertTrue(self.verify_resource_status(stack_identifier, 'bar', 'CREATE_COMPLETE'))
        resource_events = self.client.events.list(stack_identifier, 'bar')
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_replace_restrict, 2))
        self.assertTrue(self._check_for_restriction_reason(resource_events, reason_update_restrict, 0))