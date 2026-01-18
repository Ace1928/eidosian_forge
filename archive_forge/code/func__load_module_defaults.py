from __future__ import (absolute_import, division, print_function)
import itertools
import operator
import os
from copy import copy as shallowcopy
from functools import cache
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.dataloader import DataLoader
from ansible.playbook.attribute import Attribute, FieldAttribute, ConnectionFieldAttribute, NonInheritableFieldAttribute
from ansible.plugins.loader import module_loader, action_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier, get_unique_id
def _load_module_defaults(self, name, value):
    if value is None:
        return
    if not isinstance(value, list):
        value = [value]
    validated_module_defaults = []
    for defaults_dict in value:
        if not isinstance(defaults_dict, dict):
            raise AnsibleParserError('The field \'module_defaults\' is supposed to be a dictionary or list of dictionaries, the keys of which must be static action, module, or group names. Only the values may contain templates. For example: {\'ping\': "{{ ping_defaults }}"}')
        validated_defaults_dict = {}
        for defaults_entry, defaults in defaults_dict.items():
            if defaults_entry.startswith('group/'):
                group_name = defaults_entry.split('group/')[-1]
                if self.play is not None:
                    group_name, dummy = self._resolve_group(group_name)
                defaults_entry = 'group/' + group_name
                validated_defaults_dict[defaults_entry] = defaults
            else:
                if len(defaults_entry.split('.')) < 3:
                    defaults_entry = 'ansible.legacy.' + defaults_entry
                resolved_action = self._resolve_action(defaults_entry)
                if resolved_action:
                    validated_defaults_dict[resolved_action] = defaults
                if defaults_entry.startswith('ansible.legacy.'):
                    resolved_action = self._resolve_action(defaults_entry.replace('ansible.legacy.', 'ansible.builtin.'), mandatory=False)
                    if resolved_action:
                        validated_defaults_dict[resolved_action] = defaults
        validated_module_defaults.append(validated_defaults_dict)
    return validated_module_defaults