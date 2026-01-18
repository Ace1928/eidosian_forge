import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@staticmethod
def _get_expected_info(wwpns=['514f0c50023f6c00', '514f0c50023f6c01'], targets=1, remote_scan=False):
    execute_results = []
    expected_cmds = []
    for i in range(0, targets):
        expected_cmds += [mock.call(f'grep -Gil "{wwpns[i]}" /sys/class/fc_transport/target6:*/port_name', shell=True)]
        if remote_scan:
            execute_results += [('', ''), (f'/sys/class/fc_remote_ports/rport-6:0-{i + 1}/port_name\n', '')]
            expected_cmds += [mock.call(f'grep -Gil "{wwpns[i]}" /sys/class/fc_remote_ports/rport-6:*/port_name', shell=True)]
        else:
            execute_results += [(f'/sys/class/fc_transport/target6:0:{i + 1}/port_name\n', '')]
    return (execute_results, expected_cmds)