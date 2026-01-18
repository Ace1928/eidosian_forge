from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.create')
def _test_cluster_create_success(self, cmd, expected_args, expected_kwargs, mock_create, mock_get):
    mock_ct = mock.MagicMock()
    mock_ct.uuid = 'xxx'
    mock_get.return_value = mock_ct
    self._test_arg_success(cmd)
    expected = self._get_expected_args_create(*expected_args, **expected_kwargs)
    mock_create.assert_called_with(**expected)