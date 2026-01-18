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
def _get_action_handler_with_module_context(self, templar):
    """
        Returns the correct action plugin to handle the requestion task action and the module context
        """
    module_collection, separator, module_name = self._task.action.rpartition('.')
    module_prefix = module_name.split('_')[0]
    if module_collection:
        network_action = '{0}.{1}'.format(module_collection, module_prefix)
    else:
        network_action = module_prefix
    collections = self._task.collections
    module = self._shared_loader_obj.module_loader.find_plugin_with_context(self._task.action, collection_list=collections)
    if not module.resolved or not module.action_plugin:
        module = None
    if module is not None:
        handler_name = module.action_plugin
    elif self._shared_loader_obj.action_loader.has_plugin(self._task.action, collection_list=collections):
        handler_name = self._task.action
    elif all((module_prefix in C.NETWORK_GROUP_MODULES, self._shared_loader_obj.action_loader.has_plugin(network_action, collection_list=collections))):
        handler_name = network_action
        display.vvvv('Using network group action {handler} for {action}'.format(handler=handler_name, action=self._task.action), host=self._play_context.remote_addr)
    else:
        handler_name = 'ansible.legacy.normal'
        collections = None
    if any((self._connection.supports_persistence and C.USE_PERSISTENT_CONNECTIONS, self._connection.force_persistence)):
        handler_class = self._shared_loader_obj.action_loader.get(handler_name, class_only=True)
        if getattr(handler_class, '_requires_connection', True):
            self._play_context.timeout = self._connection.get_option('persistent_command_timeout')
            display.vvvv('attempting to start connection', host=self._play_context.remote_addr)
            display.vvvv('using connection plugin %s' % self._connection.transport, host=self._play_context.remote_addr)
            options = self._connection.get_options()
            socket_path = start_connection(self._play_context, options, self._task._uuid)
            display.vvvv('local domain socket path is %s' % socket_path, host=self._play_context.remote_addr)
            setattr(self._connection, '_socket_path', socket_path)
        else:
            self._connection = self._get_connection({}, templar, 'local')
    handler = self._shared_loader_obj.action_loader.get(handler_name, task=self._task, connection=self._connection, play_context=self._play_context, loader=self._loader, templar=templar, shared_loader_obj=self._shared_loader_obj, collection_list=collections)
    if not handler:
        raise AnsibleError("the handler '%s' was not found" % handler_name)
    return (handler, module)