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
def _get_template_contents(self):
    """Retrieve the contents of the parser template

        :return: The parser's contents
        :rtype: str
        """
    template_contents = None
    template_path = self._task.args.get('parser').get('template_path')
    if template_path:
        try:
            with open(template_path, 'rb') as file_handler:
                try:
                    template_contents = to_text(file_handler.read(), errors='surrogate_or_strict')
                except UnicodeError:
                    raise AnsibleActionFail('Template source files must be utf-8 encoded')
        except FileNotFoundError as exc:
            raise AnsibleActionFail("Failed to open template '{tpath}'. Error: {err}".format(tpath=template_path, err=to_native(exc)))
    return template_contents