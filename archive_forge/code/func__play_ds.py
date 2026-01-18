from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.splitter import parse_kv
from ansible.parsing.utils.yaml import from_yaml
from ansible.playbook import Playbook
from ansible.playbook.play import Play
from ansible.utils.display import Display
def _play_ds(self, pattern, async_val, poll):
    check_raw = context.CLIARGS['module_name'] in C.MODULE_REQUIRE_ARGS
    module_args_raw = context.CLIARGS['module_args']
    module_args = None
    if module_args_raw and module_args_raw.startswith('{') and module_args_raw.endswith('}'):
        try:
            module_args = from_yaml(module_args_raw.strip(), json_only=True)
        except AnsibleParserError:
            pass
    if not module_args:
        module_args = parse_kv(module_args_raw, check_raw=check_raw)
    mytask = {'action': {'module': context.CLIARGS['module_name'], 'args': module_args}, 'timeout': context.CLIARGS['task_timeout']}
    if context.CLIARGS['module_name'] not in C._ACTION_ALL_INCLUDE_ROLE_TASKS and any(frozenset((async_val, poll))):
        mytask['async_val'] = async_val
        mytask['poll'] = poll
    return dict(name='Ansible Ad-Hoc', hosts=pattern, gather_facts='no', tasks=[mytask])