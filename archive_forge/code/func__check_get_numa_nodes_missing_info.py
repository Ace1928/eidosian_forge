from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def _check_get_numa_nodes_missing_info(self):
    numa_node = mock.MagicMock()
    self._hostutils._conn.Msvm_NumaNode.return_value = [numa_node, numa_node]
    nodes_info = self._hostutils.get_numa_nodes()
    self.assertEqual([], nodes_info)