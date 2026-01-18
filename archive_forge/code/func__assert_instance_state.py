import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def _assert_instance_state(self, nested_identifier, num_complete, num_failed):
    for res in self.client.resources.list(nested_identifier):
        if 'COMPLETE' in res.resource_status:
            num_complete = num_complete - 1
        elif 'FAILED' in res.resource_status:
            num_failed = num_failed - 1
    self.assertEqual(0, num_failed)
    self.assertEqual(0, num_complete)