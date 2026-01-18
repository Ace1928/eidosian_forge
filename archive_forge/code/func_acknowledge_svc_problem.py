from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def acknowledge_svc_problem(self, host, services=None):
    """
        This command is used to acknowledge a particular
        service problem.

        By acknowledging the current problem, future notifications
        for the same servicestate are disabled

        Syntax: ACKNOWLEDGE_SVC_PROBLEM;<host_name>;<service_description>;
        <sticky>;<notify>;<persistent>;<author>;<comment>
        """
    cmd = 'ACKNOWLEDGE_SVC_PROBLEM'
    if services is None:
        services = []
    for service in services:
        ack_cmd_str = self._fmt_ack_str(cmd, host, svc=service)
        self._write_command(ack_cmd_str)