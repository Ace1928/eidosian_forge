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
@staticmethod
def _field_split(path):
    """Split the path into it's parts

        :param path: The user provided path
        :type path: str
        :return: the individual parts of the path
        :rtype: list
        """
    que = list(path)
    val = que.pop(0)
    fields = []
    try:
        while True:
            field = ''
            if val == '.':
                val = que.pop(0)
            if val == '[':
                val = que.pop(0)
                while val != ']':
                    field += val
                    val = que.pop(0)
                val = que.pop(0)
            else:
                while val not in ['.', '[']:
                    field += val
                    val = que.pop(0)
            try:
                fields.append(ast.literal_eval(field))
            except Exception:
                fields.append(re.sub('[\'"]', '', field))
    except IndexError:
        try:
            fields.append(ast.literal_eval(field))
        except Exception:
            fields.append(re.sub('[\'"]', '', field))
    return fields