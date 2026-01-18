import multiprocessing
from unittest import mock
from neutron_lib.tests import _base as base
from neutron_lib.utils import host
class TestCpuCount(base.BaseTestCase):

    @mock.patch.object(multiprocessing, 'cpu_count', return_value=7)
    def test_cpu_count(self, mock_cpu_count):
        self.assertEqual(7, host.cpu_count())
        mock_cpu_count.assert_called_once_with()

    @mock.patch.object(multiprocessing, 'cpu_count', side_effect=NotImplementedError())
    def test_cpu_count_not_implemented(self, mock_cpu_count):
        self.assertEqual(1, host.cpu_count())
        mock_cpu_count.assert_called_once_with()