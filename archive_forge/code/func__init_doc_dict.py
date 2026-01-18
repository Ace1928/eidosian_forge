from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def _init_doc_dict():
    """ initialize a return dict for docs with the expected structure """
    return {k: None for k in string_to_vars.values()}