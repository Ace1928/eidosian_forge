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
def _create_validation_task(self, argument_spec, entrypoint_name):
    """Create a new task data structure that uses the validate_argument_spec action plugin.

        :param argument_spec: The arg spec definition for a particular role entry point.
            This will be the entire arg spec for the entry point as read from the input file.
        :param entrypoint_name: The name of the role entry point associated with the
            supplied `argument_spec`.
        """
    task_name = "Validating arguments against arg spec '%s'" % entrypoint_name
    if 'short_description' in argument_spec:
        task_name = task_name + ' - ' + argument_spec['short_description']
    return {'action': {'module': 'ansible.builtin.validate_argument_spec', 'argument_spec': argument_spec.get('options', {}), 'provided_arguments': self._role_params, 'validate_args_context': {'type': 'role', 'name': self._role_name, 'argument_spec_name': entrypoint_name, 'path': self._role_path}}, 'name': task_name, 'tags': ['always']}