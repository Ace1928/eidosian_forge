from unittest import mock
from magnumclient.tests.v1 import shell_test_base
def _get_expected_args_create(self, project_id, resource, hard_limit):
    expected_args = {}
    expected_args['project_id'] = project_id
    expected_args['resource'] = resource
    expected_args['hard_limit'] = hard_limit
    return expected_args