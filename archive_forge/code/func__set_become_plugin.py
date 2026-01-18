from __future__ import (absolute_import, division, print_function)
import os
import pty
import time
import json
import signal
import subprocess
import sys
import termios
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleConnectionFailure, AnsibleActionFail, AnsibleActionSkip
from ansible.executor.task_result import TaskResult
from ansible.executor.module_common import get_action_args_with_defaults
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.connection import write_to_file_descriptor
from ansible.playbook.conditional import Conditional
from ansible.playbook.task import Task
from ansible.plugins import get_plugin_class
from ansible.plugins.loader import become_loader, cliconf_loader, connection_loader, httpapi_loader, netconf_loader, terminal_loader
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, isidentifier
def _set_become_plugin(self, cvars, templar, connection):
    if cvars.get('ansible_become') is not None:
        become = boolean(templar.template(cvars['ansible_become']))
    else:
        become = self._task.become
    if become:
        if cvars.get('ansible_become_method'):
            become_plugin = self._get_become(templar.template(cvars['ansible_become_method']))
        else:
            become_plugin = self._get_become(self._task.become_method)
    else:
        become_plugin = None
    try:
        connection.set_become_plugin(become_plugin)
    except AttributeError:
        pass
    if become_plugin:
        if getattr(connection.become, 'require_tty', False) and (not getattr(connection, 'has_tty', False)):
            raise AnsibleError("The '%s' connection does not provide a TTY which is required for the selected become plugin: %s." % (connection._load_name, become_plugin.name))
        self._play_context.set_become_plugin(become_plugin.name)