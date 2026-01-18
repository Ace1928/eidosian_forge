from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Container, Mapping, Set, Sequence
from types import MappingProxyType
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, text_type
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.role.metadata import RoleMetadata
from ansible.playbook.taggable import Taggable
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.path import is_subpath
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars
def _load_role_data(self, role_include, parent_role=None):
    self._role_name = role_include.role
    self._role_path = role_include.get_role_path()
    self._role_collection = role_include._role_collection
    self._role_params = role_include.get_role_params()
    self._variable_manager = role_include.get_variable_manager()
    self._loader = role_include.get_loader()
    if parent_role:
        self.add_parent(parent_role)
    for attr_name in self.fattributes:
        setattr(self, f'_{attr_name}', getattr(role_include, f'_{attr_name}', Sentinel))
    self._role_vars = self._load_role_yaml('vars', main=self._from_files.get('vars'), allow_dir=True)
    if self._role_vars is None:
        self._role_vars = {}
    elif not isinstance(self._role_vars, Mapping):
        raise AnsibleParserError("The vars/main.yml file for role '%s' must contain a dictionary of variables" % self._role_name)
    self._default_vars = self._load_role_yaml('defaults', main=self._from_files.get('defaults'), allow_dir=True)
    if self._default_vars is None:
        self._default_vars = {}
    elif not isinstance(self._default_vars, Mapping):
        raise AnsibleParserError("The defaults/main.yml file for role '%s' must contain a dictionary of variables" % self._role_name)
    metadata = self._load_role_yaml('meta')
    if metadata:
        self._metadata = RoleMetadata.load(metadata, owner=self, variable_manager=self._variable_manager, loader=self._loader)
        self._dependencies = self._load_dependencies()
    self.collections = []
    if self._role_collection:
        self.collections.insert(0, self._role_collection)
    else:
        default_collection = AnsibleCollectionConfig.default_collection
        if default_collection:
            self.collections.insert(0, default_collection)
        add_all_plugin_dirs(self._role_path)
    if self._metadata.collections:
        self.collections.extend((c for c in self._metadata.collections if c not in self.collections))
    if self.collections:
        default_append_collection = 'ansible.builtin' if self._role_collection else 'ansible.legacy'
        if 'ansible.builtin' not in self.collections and 'ansible.legacy' not in self.collections:
            self.collections.append(default_append_collection)
    task_data = self._load_role_yaml('tasks', main=self._from_files.get('tasks'))
    if self._should_validate:
        role_argspecs = self._get_role_argspecs()
        task_data = self._prepend_validation_task(task_data, role_argspecs)
    if task_data:
        try:
            self._task_blocks = load_list_of_blocks(task_data, play=self._play, role=self, loader=self._loader, variable_manager=self._variable_manager)
        except AssertionError as e:
            raise AnsibleParserError("The tasks/main.yml file for role '%s' must contain a list of tasks" % self._role_name, obj=task_data, orig_exc=e)
    handler_data = self._load_role_yaml('handlers', main=self._from_files.get('handlers'))
    if handler_data:
        try:
            self._handler_blocks = load_list_of_blocks(handler_data, play=self._play, role=self, use_handlers=True, loader=self._loader, variable_manager=self._variable_manager)
        except AssertionError as e:
            raise AnsibleParserError("The handlers/main.yml file for role '%s' must contain a list of tasks" % self._role_name, obj=handler_data, orig_exc=e)