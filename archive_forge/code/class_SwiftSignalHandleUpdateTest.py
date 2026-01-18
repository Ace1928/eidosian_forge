from heat_integrationtests.functional import functional_base
class SwiftSignalHandleUpdateTest(functional_base.FunctionalTestsBase):

    def test_stack_update_same_template_replace_no_url(self):
        if not self.is_service_available('object-store'):
            self.skipTest('object-store service not available, skipping')
        stack_identifier = self.stack_create(template=test_template)
        stack = self.client.stacks.get(stack_identifier)
        orig_url = self._stack_output(stack, 'signal_url')
        orig_curl = self._stack_output(stack, 'signal_curl')
        self.update_stack(stack_identifier, test_template)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(orig_url, self._stack_output(stack, 'signal_url'))
        self.assertEqual(orig_curl, self._stack_output(stack, 'signal_curl'))