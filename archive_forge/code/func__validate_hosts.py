from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.helpers import load_list_of_blocks, load_list_of_roles
from ansible.playbook.role import Role, hash_params
from ansible.playbook.task import Task
from ansible.playbook.taggable import Taggable
from ansible.vars.manager import preprocess_vars
from ansible.utils.display import Display
def _validate_hosts(self, attribute, name, value):
    if 'hosts' in self._ds:
        if not value:
            raise AnsibleParserError('Hosts list cannot be empty. Please check your playbook')
        if is_sequence(value):
            for entry in value:
                if entry is None:
                    raise AnsibleParserError("Hosts list cannot contain values of 'None'. Please check your playbook")
                elif not isinstance(entry, (binary_type, text_type)):
                    raise AnsibleParserError("Hosts list contains an invalid host value: '{host!s}'".format(host=entry))
        elif not isinstance(value, (binary_type, text_type)):
            raise AnsibleParserError('Hosts list must be a sequence or string. Please check your playbook.')