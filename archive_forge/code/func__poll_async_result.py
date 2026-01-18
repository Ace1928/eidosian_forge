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
def _poll_async_result(self, result, templar, task_vars=None):
    """
        Polls for the specified JID to be complete
        """
    if task_vars is None:
        task_vars = self._job_vars
    async_jid = result.get('ansible_job_id')
    if async_jid is None:
        return dict(failed=True, msg='No job id was returned by the async task')
    async_task = Task.load(dict(action='async_status', args={'jid': async_jid}, check_mode=self._task.check_mode, environment=self._task.environment))
    async_handler = self._shared_loader_obj.action_loader.get('ansible.legacy.async_status', task=async_task, connection=self._connection, play_context=self._play_context, loader=self._loader, templar=templar, shared_loader_obj=self._shared_loader_obj)
    time_left = self._task.async_val
    while time_left > 0:
        time.sleep(self._task.poll)
        try:
            async_result = async_handler.run(task_vars=task_vars)
            if int(async_result.get('finished', 0)) == 1 or ('failed' in async_result and async_result.get('_ansible_parsed', False)) or 'skipped' in async_result:
                break
        except Exception as e:
            display.vvvv('Exception during async poll, retrying... (%s)' % to_text(e))
            display.debug('Async poll exception was:\n%s' % to_text(traceback.format_exc()))
            try:
                async_handler._connection.reset()
            except AttributeError:
                pass
            time_left -= self._task.poll
            if time_left <= 0:
                raise
        else:
            time_left -= self._task.poll
            self._final_q.send_callback('v2_runner_on_async_poll', TaskResult(self._host.name, async_task._uuid, async_result, task_fields=async_task.dump_attrs()))
    if int(async_result.get('finished', 0)) != 1:
        if async_result.get('_ansible_parsed'):
            return dict(failed=True, msg='async task did not complete within the requested time - %ss' % self._task.async_val, async_result=async_result)
        else:
            return dict(failed=True, msg='async task produced unparseable results', async_result=async_result)
    else:
        cleanup_task = Task.load({'async_status': {'jid': async_jid, 'mode': 'cleanup'}, 'check_mode': self._task.check_mode, 'environment': self._task.environment})
        cleanup_handler = self._shared_loader_obj.action_loader.get('ansible.legacy.async_status', task=cleanup_task, connection=self._connection, play_context=self._play_context, loader=self._loader, templar=templar, shared_loader_obj=self._shared_loader_obj)
        cleanup_handler.run(task_vars=task_vars)
        cleanup_handler.cleanup(force=True)
        async_handler.cleanup(force=True)
        return async_result