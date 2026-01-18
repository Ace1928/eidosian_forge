from __future__ import absolute_import, division, print_function
import ast
import re
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.action import ActionBase
from jinja2 import Template, TemplateSyntaxError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.update_fact import DOCUMENTATION
def _ensure_valid_jinja(self):
    """Ensure each path is jinja valid"""
    errors = []
    for entry in self._task.args['updates']:
        try:
            Template('{{' + entry['path'] + '}}')
        except TemplateSyntaxError as exc:
            error = "While processing '{path}' found malformed path. Ensure syntax follows valid jinja format. The error was: {error}".format(path=entry['path'], error=to_native(exc))
            errors.append(error)
    if errors:
        raise AnsibleActionFail(' '.join(errors))