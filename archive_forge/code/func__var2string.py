from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def _var2string(value):
    """ reverse lookup of the dict above """
    for k, v in string_to_vars.items():
        if v == value:
            return k