import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class CreateStackTest(functional_base.FunctionalTestsBase):

    def test_create_rollback(self):
        values = {'fail': True, 'value': 'test_create_rollback'}
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], values)
        self.stack_create(template=template, expected_status='ROLLBACK_COMPLETE', disable_rollback=False)