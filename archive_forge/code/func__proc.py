from __future__ import absolute_import, division, print_function
import syslog
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def _proc(*a):
    return a