from __future__ import absolute_import, division, print_function
import json
from importlib import import_module
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible.module_utils.connection import ConnectionError as AnsibleConnectionError
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.cli_parse import DOCUMENTATION
def _update_template_path(self, template_extension):
    """Update the template_path in the task args
        If not provided, generate template name using os and command

        :param template_extension: The parser specific template extension
        :type template extension: str
        """
    if not self._task.args.get('parser').get('template_path'):
        if self._task.args.get('parser').get('os'):
            oper_sys = self._task.args.get('parser').get('os')
        else:
            oper_sys = self._os_from_task_vars()
        cmd_as_fname = self._task.args.get('parser').get('command').replace(' ', '_')
        fname = '{os}_{cmd}.{ext}'.format(os=oper_sys, cmd=cmd_as_fname, ext=template_extension)
        source = self._find_needle('templates', fname)
        self._debug('template_path in task args updated to {source}'.format(source=source))
        self._task.args['parser']['template_path'] = source