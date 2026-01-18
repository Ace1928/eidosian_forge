from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
def _get_expected_args_list(self, limit=None, sort_dir=None, sort_key=None, detail=False):
    expected_args = {}
    expected_args['limit'] = limit
    expected_args['sort_dir'] = sort_dir
    expected_args['sort_key'] = sort_key
    expected_args['detail'] = detail
    return expected_args